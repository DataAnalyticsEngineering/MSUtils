from collections import defaultdict
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from MSUtils.voronoi import VoronoiImage, VoronoiTessellation


class PeriodicVoronoiImageErosion:
    def __init__(
        self,
        voroImg: VoronoiImage,
        voroTess: VoronoiTessellation,
        interface_thickness: float = 0.05,
    ) -> Self:
        """
        Class to handle the Erosion of Periodic Voronoi Images.

        Args:
            voroImg (VoronoiImage): Image.
            voroTess (VoronoiTessalation): Voronoi-Tesselation
            interface_thickness (float, optional): Total thickness to extrude to. Defaults to 5.

        Returns:
            Self: Object
        """
        self.image = voroImg.image
        self.seeds = voroTess.seeds
        self.extrusion_factor = (
            interface_thickness / 2
        )  # Extrusion factor in both directions of the plane
        self.L = np.array(voroImg.L)
        self.eroded_image = None
        self.N = np.array(voroImg.resolution)
        self.voroTess = voroTess

        self.num_crystals = len(np.unique(self.image))
        self.ridge_metadata = {}  # tag  →  (normal, ia, ib)

        self._precompute_polyhedrons(voroTess)
        self._shrink_and_analyze()

    def _periodic_thickness_consistent_erosion(self, crystal) -> npt.ArrayLike:
        """
        Returns a Boolean mask the same shape as self.image:
        - True  → voxel still belongs to `crystal` after erosion
        - False → voxel was removed (belongs to a grain boundary)
        """
        crystal_mask = self.image == crystal
        crystal_idx = np.array(np.where(crystal_mask)).T
        voxel_coords = (crystal_idx + 0.5) * self.voxel_scale

        for poly_index, side_flag in self.polytrack.get(crystal, []):
            delaunay = (
                self.polylist_A[poly_index]
                if side_flag == 0
                else self.polylist_B[poly_index]
            )

            # outward-pointing normal for this crystal
            i_idx, j_idx = self.polyinfo[poly_index]
            diff = (
                self.voroTess.voronoi.points[j_idx]
                - self.voroTess.voronoi.points[i_idx]
            )
            normal = diff / np.linalg.norm(diff)
            if side_flag:  # current crystal is j ⇒ flip
                normal = -normal

            # bounding-box pre-filter
            min_xyz, max_xyz = (self.bbox_A if side_flag == 0 else self.bbox_B)[
                poly_index
            ]
            bbox_mask = np.all(
                (voxel_coords >= min_xyz) & (voxel_coords <= max_xyz), axis=1
            )
            cand_ids = np.where(bbox_mask)[0]
            if cand_ids.size == 0:
                continue
            cand_coords = voxel_coords[cand_ids]

            # STEP-1: keep voxels within ±extrusion_factor of ridge plane
            d = (cand_coords - delaunay.points[0]) @ normal
            band_mask = np.abs(d) <= self.extrusion_factor
            if not np.any(band_mask):
                continue
            band_ids = cand_ids[band_mask]
            band_coords = cand_coords[band_mask]

            # STEP-2: 2-D point-in-polygon test in the ridge plane
            polygon = delaunay.points[:-1] - delaunay.points[0]
            poly_roll = np.roll(polygon, -1, axis=0)
            edges = (poly_roll - polygon)[np.newaxis, :, :]

            proj = band_coords - delaunay.points[0]
            proj -= np.outer((proj @ normal), normal)  # project
            vecs = proj[:, np.newaxis, :] - polygon[np.newaxis, :, :]
            cross = np.cross(edges, vecs, axis=-1)
            ref = cross[:, :1, :]
            same_dir = np.sum(cross * ref, axis=-1)
            inside_poly = np.logical_or(
                np.all(same_dir >= 0, 1), np.all(same_dir <= 0, 1)
            )
            if not np.any(inside_poly):
                continue

            # remove voxels
            kill_ids = band_ids[inside_poly]
            kill_idx = crystal_idx[kill_ids]
            crystal_mask[tuple(kill_idx.T)] = False

        return crystal_mask

    def _shrink_and_analyze(self) -> None:
        """
        Builds:
        • self.eroded_image        – crystals (0…N-1) and ridge IDs (N…)
        • self.coords_array …      – unchanged from your previous implementation
        • self.ridge_metadata[tag] – (normal, crystal_a, crystal_b)
        """
        N = self.num_crystals
        unique_crystals = np.arange(N)  # 0 … N-1
        eroded_image = -1 * np.ones_like(self.image)
        Nx, Ny, Nz = self.image.shape
        hx, hy, hz = self.L / np.array([Nx, Ny, Nz])
        self.voxel_scale = np.array([hx, hy, hz])

        for crystal in unique_crystals:
            crystal_mask = self.image == crystal
            # interior voxels that survive the erosion step
            interior_mask = self._periodic_thickness_consistent_erosion(crystal)

            # voxels that WERE removed => potential grain-boundary voxels
            boundary_mask = crystal_mask & np.logical_not(interior_mask)
            boundary_idx = np.array(np.where(boundary_mask)).T
            if boundary_idx.size == 0:
                # crystal disappeared completely (rare)
                continue
            voxel_coords = (boundary_idx + 0.5) * self.voxel_scale
            voxel_checked = np.zeros(len(voxel_coords), dtype=bool)

            for poly_index, side_flag in self.polytrack.get(crystal, []):
                tag = self.ridge_tags[poly_index]  # N + poly_index
                delaunay = (
                    self.polylist_A[poly_index]
                    if side_flag == 0
                    else self.polylist_B[poly_index]
                )

                min_xyz, max_xyz = (self.bbox_A if side_flag == 0 else self.bbox_B)[
                    poly_index
                ]
                bbox_mask = np.all(
                    (voxel_coords >= min_xyz) & (voxel_coords <= max_xyz), axis=1
                )
                cand_ids = np.where(bbox_mask & ~voxel_checked)[0]
                if cand_ids.size == 0:
                    continue
                cand_coords = voxel_coords[cand_ids]

                inside = delaunay.find_simplex(cand_coords) >= 0
                if not np.any(inside):
                    continue
                in_ids = cand_ids[inside]
                in_idx = boundary_idx[in_ids]

                # write ridge tag into eroded_image (skip if already tagged)
                new_vox = eroded_image[tuple(in_idx.T)] == -1
                in_idx = in_idx[new_vox]
                if in_idx.size == 0:
                    continue
                eroded_image[tuple(in_idx.T)] = tag

                # ---------- outward normal (store once per tag) ---------------
                if tag not in self.ridge_metadata:
                    i_idx, j_idx = self.polyinfo[poly_index]
                    diff = (
                        self.voroTess.voronoi.points[j_idx]
                        - self.voroTess.voronoi.points[i_idx]
                    )
                    normal = diff / np.linalg.norm(diff)
                    cA = self.voroTess.crystal_index_map[i_idx]
                    cB = self.voroTess.crystal_index_map[j_idx]
                    self.ridge_metadata[tag] = (normal, cA, cB)
                # ----------------------------------------------------------------

                voxel_checked[in_ids] = True

            # crystal interior stays labelled with the crystal index
            eroded_image[(crystal_mask & interior_mask)] = crystal

        self.eroded_image = eroded_image

    def _precompute_polyhedrons(self, voroTess: VoronoiTessellation) -> None:
        self.polylist_A, self.polylist_B = [], []
        self.polyinfo, self.ridge_tags = [], []
        self.bbox_A, self.bbox_B = [], []
        self.polytrack = defaultdict(list)

        # helper maps
        self._key2tag = {}  # (cA,cB,n_key) → tag
        self.next_tag = self.num_crystals  # first ridge label
        Q = 1_000_000.0  # 1 µrad normal quantisation

        for ridge, ridge_pts in zip(
            voroTess.voronoi.ridge_vertices, voroTess.voronoi.ridge_points, strict=False
        ):
            if -1 in ridge:  # infinite ridge
                continue

            cA = self.voroTess.crystal_index_map[ridge_pts[0]]
            cB = self.voroTess.crystal_index_map[ridge_pts[1]]

            # ---------- canonical, orientation-agnostic normal ---------------
            seed_i, seed_j = voroTess.voronoi.points[ridge_pts]
            n = seed_j - seed_i
            n /= np.linalg.norm(n)
            if (
                (n[0] < 0)
                or (np.isclose(n[0], 0) and n[1] < 0)
                or (np.isclose(n[0], 0) and np.isclose(n[1], 0) and n[2] < 0)
            ):
                n = -n
            n_key = tuple((n * Q).round().astype(int))
            # -----------------------------------------------------------------

            tag = self._key2tag.setdefault(
                (min(cA, cB), max(cA, cB), n_key), self.next_tag
            )
            if tag == self.next_tag:
                self.next_tag += 1

            # -------------- build the two half-polyhedra ---------------------
            verts = voroTess.voronoi.vertices[ridge]
            poly_i = Delaunay(np.vstack((verts, seed_i)))
            poly_j = Delaunay(np.vstack((verts, seed_j)))

            idx = len(self.polylist_A)
            self.polylist_A.append(poly_i)
            self.polylist_B.append(poly_j)
            self.polyinfo.append(tuple(ridge_pts))
            self.ridge_tags.append(tag)

            pts_i = np.vstack((verts, seed_i))
            pts_j = np.vstack((verts, seed_j))
            self.bbox_A.append((pts_i.min(0), pts_i.max(0)))
            self.bbox_B.append((pts_j.min(0), pts_j.max(0)))

            self.polytrack[cA].append((idx, 0))  # hull A faces cA
            self.polytrack[cB].append((idx, 1))  # hull B faces cB)

    def write_h5(self, filepath: Path, grp_name: str, order: str = "zyx") -> None:
        """
        Write this eroded periodic voronoi image into a h5-file.

        Args:
            filepath (Path): Path to h5 file.
            grp_name (str): Name of h5py group.
            order (str, optional): Either one of 'xyz' or 'zyx'. Defaults to "zyx".
        """
        with h5py.File(filepath, "a") as h5_file:
            grp = h5_file.require_group(grp_name)
            compression_opts = 6

            # Define dtype for the ridge metadata to include ridge tag
            ridge_metadata_dtype = np.dtype(
                [
                    ("ridge_tag", "i8"),
                    ("normal", "f8", (3,)),
                    ("crystalA", "i8"),
                    ("crystalB", "i8"),
                ]
            )
            # Convert ridge_metadata dictionary to structured array
            ridge_tags = sorted(self.ridge_metadata.keys())
            ridge_metadata = np.array(
                [
                    (
                        tag,
                        self.ridge_metadata[tag][0],
                        self.ridge_metadata[tag][1],
                        self.ridge_metadata[tag][2],
                    )
                    for tag in ridge_tags
                ],
                dtype=ridge_metadata_dtype,
            )

            # Create a new field for normals, taking shape from the original image
            Nx, Ny, Nz = self.eroded_image.shape
            normals_field = np.zeros((Nx, Ny, Nz, 3))
            for i, j, k in np.ndindex(self.eroded_image.shape):
                mat_index = self.eroded_image[i, j, k]
                if mat_index >= self.num_crystals:
                    normal, _, _ = self.ridge_metadata[mat_index]
                    normals_field[i, j, k] = normal

            # Optionally permute the eroded_image before saving, based on the order parameter
            if order == "xyz":
                #################################
                # Image is in order of x, y, z
                # Vector field is in order of x, y, z
                #################################
                permuted_eroded_image = self.eroded_image
                permuted_normals_field = normals_field
            elif order == "zyx":
                #################################
                # Image is in order of z, y, x
                # Vector field is in order of x, y, z
                #################################
                permuted_eroded_image = self.eroded_image.transpose(2, 1, 0)
                permuted_normals_field = normals_field.transpose(2, 1, 0, 3)

            # Save eroded image to .h5 file
            if "eroded_image" in grp:
                del grp["eroded_image"]
                print("Overwriting existing 'eroded_image' dataset.")
            image_dataset = grp.create_dataset(
                "eroded_image",
                data=permuted_eroded_image,
                dtype=np.int32,
                compression="gzip",
                compression_opts=compression_opts,
            )
            image_dataset.attrs["permute_order"] = order
            image_dataset.attrs["interface_thickness"] = self.extrusion_factor * 2
            image_dataset.attrs["L"] = self.L
            image_dataset.attrs.create(
                "VoronoiSeeds_xyz", np.array(self.seeds, dtype=np.float64)
            )

            # Save ridge metadata to .h5 file
            if "ridge_metadata" in grp:
                del grp["ridge_metadata"]
                print("Overwriting existing 'ridge_metadata' dataset.")
            ridge_dataset = grp.create_dataset(
                "ridge_metadata",
                data=ridge_metadata,
                compression="gzip",
                compression_opts=compression_opts,
            )

            # Save normals to .h5 file
            if "normals" in grp:
                del grp["normals"]
                print("Overwriting existing 'normals' dataset.")
            normals_dataset = grp.create_dataset(
                "normals",
                data=permuted_normals_field,
                dtype="f8",
                compression="gzip",
                compression_opts=compression_opts,
            )
            normals_dataset.attrs["permute_order"] = order
