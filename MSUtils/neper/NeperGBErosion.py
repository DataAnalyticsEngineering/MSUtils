import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np

from MSUtils.general.grid import image_in_order
from MSUtils.neper.NeperMicrostructure import NeperMicrostructure, _read_faces


class NeperGBErosion:
    """Rasterize finite-thickness grain boundaries from Neper face geometry."""

    def __init__(
        self,
        microstructure: NeperMicrostructure,
        interface_thickness: float,
    ):
        if not np.isfinite(interface_thickness) or interface_thickness < 0:
            raise ValueError("interface_thickness must be finite and nonnegative.")

        self.image = microstructure.image
        self.grid = microstructure.grid
        self.L = np.asarray(self.grid.lengths)
        self.voxel_size = np.asarray(microstructure.voxel_size, dtype=float)
        grain_offset = int(microstructure.void_present)
        self.label_offset = grain_offset - 1
        self.num_crystals = microstructure.grain_count + grain_offset
        self.rotation_matrices = microstructure.rotation_matrices
        self.interface_thickness = float(interface_thickness)
        self.eroded_image = self.image.copy()
        face_filename = microstructure.tesr_filename.with_suffix(".stface")
        self._rasterize(_read_faces(face_filename))

    @staticmethod
    def _canonical_normal(normal: np.ndarray) -> np.ndarray:
        for component in normal:
            if not np.isclose(component, 0.0):
                return normal if component > 0 else -normal
        raise ValueError("A Neper face has no well-defined normal.")

    def _face_normal(
        self, poly_a: int, poly_b: int, vertices: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        _, _, basis = np.linalg.svd(
            vertices - vertices.mean(axis=0), full_matrices=False
        )
        normal = self._canonical_normal(basis[-1])

        grain_a, grain_b = sorted(
            (poly_a + self.label_offset, poly_b + self.label_offset)
        )

        return normal, grain_a, grain_b

    def _periodic_faces(self, vertices: np.ndarray):
        half = self.interface_thickness / 2
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        shifts = []
        for axis in range(3):
            first = int(np.ceil((-half - maximum[axis]) / self.L[axis]))
            last = int(np.floor((self.L[axis] + half - minimum[axis]) / self.L[axis]))
            shifts.append(range(first, last + 1))
        for shift in product(*shifts):
            yield vertices + np.asarray(shift) * self.L

    def _paint_face(
        self,
        vertices: np.ndarray,
        normal: np.ndarray,
        grain_a: int,
        grain_b: int,
        tag: int,
        best_distance: np.ndarray,
    ) -> None:
        half = self.interface_thickness / 2
        lower = np.ceil((vertices.min(axis=0) - half) / self.voxel_size - 0.5).astype(
            int
        )
        upper = np.floor((vertices.max(axis=0) + half) / self.voxel_size - 0.5).astype(
            int
        )
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(self.image.shape) - 1)
        if np.any(lower > upper):
            return

        polygon = vertices
        edges = np.roll(polygon, -1, axis=0) - polygon
        tolerance = 1e-10 * max(self.L)
        x_indices = np.arange(lower[0], upper[0] + 1)
        y_indices = np.arange(lower[1], upper[1] + 1)
        xx, yy = np.meshgrid(x_indices, y_indices, indexing="ij")
        xy = np.column_stack((xx.ravel(), yy.ravel()))

        for z_index in range(lower[2], upper[2] + 1):
            indices = np.column_stack((xy, np.full(len(xy), z_index, dtype=np.int64)))
            labels = self.image[tuple(indices.T)]
            relevant = (labels == grain_a) | (labels == grain_b)
            if not np.any(relevant):
                continue
            indices = indices[relevant]
            coordinates = (indices + 0.5) * self.voxel_size
            distance = (coordinates - polygon[0]) @ normal
            band = np.abs(distance) <= half + tolerance
            if not np.any(band):
                continue
            indices = indices[band]
            distance = distance[band]
            coordinates = coordinates[band] - distance[:, None] * normal

            vectors = coordinates[:, None, :] - polygon[None, :, :]
            signs = np.cross(edges[None, :, :], vectors) @ normal
            inside = np.all(signs >= -tolerance, axis=1) | np.all(
                signs <= tolerance, axis=1
            )
            if not np.any(inside):
                continue
            indices = indices[inside]
            distance = np.abs(distance[inside])
            current = best_distance[tuple(indices.T)]
            closer = distance < current
            indices = indices[closer]
            if len(indices):
                self.eroded_image[tuple(indices.T)] = tag
                best_distance[tuple(indices.T)] = distance[closer]

    def _rasterize(self, faces) -> None:
        tags = {}
        metadata = {}
        best_distance = np.full(self.image.shape, np.inf, dtype=np.float32)

        for poly_a, poly_b, vertices in faces:
            normal, grain_a, grain_b = self._face_normal(poly_a, poly_b, vertices)
            key = (grain_a, grain_b, *np.round(normal * 1_000_000).astype(int))
            tag = tags.setdefault(key, self.num_crystals + len(tags))
            metadata[tag] = (normal, grain_a, grain_b)
            for periodic_vertices in self._periodic_faces(vertices):
                self._paint_face(
                    periodic_vertices,
                    normal,
                    grain_a,
                    grain_b,
                    tag,
                    best_distance,
                )

        active = np.unique(self.eroded_image[self.eroded_image >= self.num_crystals])
        mapping = {old: self.num_crystals + i for i, old in enumerate(active)}
        for old, new in mapping.items():
            if old != new:
                self.eroded_image[self.eroded_image == old] = new
        self.ridge_metadata = {mapping[tag]: metadata[tag] for tag in active}

    def write_h5(
        self,
        h5_filename: str | Path,
        grp_name: str,
        order: str = "zyx",
        save_normals: bool = False,
        save_orientations: bool = False,
    ) -> None:
        """Write the eroded image and grain-boundary metadata."""
        grid_attributes = self.grid.to_h5_attributes(order)
        orientation_names = tuple(f"eroded_image_crystal_axis_{axis}" for axis in "xyz")
        with h5py.File(h5_filename, "a") as h5_file:
            group = h5_file.require_group(grp_name)
            for name in (
                "eroded_image",
                "eroded_image_normals",
                "rotation_matrices",
                *orientation_names,
            ):
                if name in group:
                    del group[name]

            dataset = group.create_dataset(
                "eroded_image",
                data=image_in_order(self.eroded_image, order),
                dtype=np.int32,
                compression="gzip",
                compression_opts=6,
            )
            dataset.attrs.update(
                {
                    **grid_attributes,
                    "interface_thickness": self.interface_thickness,
                    "GBVoxelInfo": json.dumps(
                        {
                            str(tag): {
                                "GB_tag": int(tag),
                                "GB_normal": normal.tolist(),
                            }
                            for tag, (normal, _, _) in sorted(
                                self.ridge_metadata.items()
                            )
                        }
                    ),
                    "num_crystals": self.num_crystals,
                    "num_GB": len(self.ridge_metadata),
                }
            )
            group.create_dataset("rotation_matrices", data=self.rotation_matrices)
            if save_orientations:
                grain_voxels = self.eroded_image < self.num_crystals
                for axis, name in enumerate(orientation_names):
                    vectors = np.zeros(self.eroded_image.shape + (3,), dtype=np.float32)
                    vectors[grain_voxels] = self.rotation_matrices[
                        self.eroded_image[grain_voxels], :, axis
                    ]
                    if order == "zyx":
                        vectors = vectors.transpose(2, 1, 0, 3)
                    dataset = group.create_dataset(
                        name,
                        data=vectors,
                        compression="gzip",
                        compression_opts=6,
                    )
                    dataset.attrs.update(grid_attributes)
            if save_normals:
                normals = np.zeros(self.eroded_image.shape + (3,), dtype=np.float64)
                for tag, (normal, _, _) in self.ridge_metadata.items():
                    normals[self.eroded_image == tag] = normal
                if order == "zyx":
                    normals = normals.transpose(2, 1, 0, 3)
                dataset = group.create_dataset(
                    "eroded_image_normals",
                    data=normals,
                    compression="gzip",
                    compression_opts=6,
                )
                dataset.attrs.update(grid_attributes)
