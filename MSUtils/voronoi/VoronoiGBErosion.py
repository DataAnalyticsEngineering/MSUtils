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
        self.extrusion_factor = interface_thickness / 2    # Extrusion factor in both directions of the plane
        self.L = np.array(voroImg.L)
        self.eroded_image = None
        self.N = np.array(voroImg.resolution)
        self.voroTess = voroTess

        # Initialize arrays to store grain boundary information
        self.coords_array = None
        self.elem_xyz_array = None
        self.materials_array = None
        self.normal_array = None

        self._precompute_polyhedrons(voroTess)
        self._shrink_and_analyze()

    def _batch_points_in_convex_polygon(self, polygon, points):
        A = polygon
        B = np.roll(polygon, -1, axis=0)
        edges = B - A          # shape (N, 2)

        # Reshape points for broadcasting
        P = points[:, np.newaxis, :]  # shape (M, 1, 2)
        A = A[np.newaxis, :, :]       # shape (1, N, 2)
        E = edges[np.newaxis, :, :]   # shape (1, N, 2)

        # Vector from A to each point
        AP = P - A                    # shape (M, N, 2)

        # Cross product for each point with each edge
        cross = np.cross(E, AP, axis=-1)
        ref = cross[:,0:1,:]

        # Check whether they all have same direction
        same_direction_mask = np.sum(cross * ref, axis=-1)

        # For each point, check if all cross products have same sign
        all_positive = np.all(same_direction_mask >= 0, axis=1)
        all_negative = np.all(same_direction_mask <= 0, axis=1)

        return np.logical_or(all_positive, all_negative)

    def _periodic_thickness_consistent_erosion(self, crystal) -> npt.ArrayLike:

        # Check if crystal touches itself (crystal is its own neighbor, requires additional tests below)
        self_touching = crystal in self.voroTess.neighbors[crystal]

        crystal_mask = (self.image == crystal)
        crystal_mask_indices = np.array(np.where(crystal_mask)).T
        voxel_coords = (crystal_mask_indices + 0.5) * self.voxel_scale

        eroded_mask = crystal_mask.copy()

        # Use the crystal label as a key in the dictionary
        poly_indices = self.polytrack.get(crystal, [])

        for poly_index in poly_indices:
            # Identify normal
            ridge_pts = self.polyinfo[poly_index]  # Seed indices
            diff = (
                self.voroTess.voronoi.points[ridge_pts[1]]
                - self.voroTess.voronoi.points[ridge_pts[0]]
            )
            norm = np.linalg.norm(diff)
            if norm == 0:
                continue  # Avoid division by zero
            normal = diff / norm

            delaunay = self.polylist[poly_index]
            points = delaunay.points

            # Bounding box filtering
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            bbox_mask = np.all(
                (voxel_coords >= min_coords) & (voxel_coords <= max_coords), axis=1
            )
            candidate_voxel_coords = voxel_coords[bbox_mask]
            candidate_indices_in_voxel_coords = np.where(bbox_mask)[0]

            if candidate_voxel_coords.size == 0:
                continue  # Skip if no candidate voxels

            # Identify grain boundary voxels
            # Keep all voxels that have less than extrusion factor distance to interface plane (projection onto normal vector)
            inside = (np.abs((candidate_voxel_coords - points[0][None,:]) @ normal) <= self.extrusion_factor)
            # Keep all voxels where the in-plane projection lies within the GB segment
            inside &= self._batch_points_in_convex_polygon(points[:-2] - points[0][None,:], ((candidate_voxel_coords - points[0][None,:]) @ (np.eye(3,3) - np.outer(normal,normal))))
            
            if self_touching:
                # Keep all crystal voxels where projection onto the facet plane ALSO lies within the facet
                inside &= delaunay.find_simplex(points[0][None,:] + (candidate_voxel_coords - points[0][None,:]) @ (np.eye(3,3) - np.outer(normal,normal))) >= 0

            if not np.any(inside):
                continue  # Skip if no inside voxels

            inside_indices_in_voxel_coords = candidate_indices_in_voxel_coords[inside]
            inside_indices = crystal_mask_indices[inside_indices_in_voxel_coords]

            eroded_mask[*inside_indices.T] = 0

        return eroded_mask

    def _shrink_and_analyze(self) -> None:
        unique_crystals = np.unique(self.image)
        eroded_image = -1 * np.ones_like(self.image)
        Nx, Ny, Nz = self.image.shape
        hx, hy, hz = self.L / np.array([Nx, Ny, Nz])

        # Initialize lists for concatenation
        coords_list = []
        elem_xyz_list = []
        materials_list = []
        normal_list = []

        # Precompute voxel scales
        self.voxel_scale = np.array([hx, hy, hz])

        for crystal in unique_crystals:
            crystal_mask = self.image == crystal
            eroded_mask = self._periodic_thickness_consistent_erosion(crystal)

            boundary_mask = np.logical_and(crystal_mask, np.logical_not(eroded_mask))
            boundary_mask_indices = np.array(np.where(boundary_mask)).T  # Shape: (num_voxels, 3)
            voxel_coords = (boundary_mask_indices + 0.5) * self.voxel_scale  # Shape: (num_voxels, 3)

            if boundary_mask_indices.size == 0:
                continue  # Skip if no boundary voxels

            # Initialize a boolean array to keep track of checked voxels
            voxel_checked = np.full(len(voxel_coords), False, dtype=bool)

            # Use the crystal label as a key in the dictionary
            poly_indices = self.polytrack.get(crystal, [])

            for poly_index in poly_indices:
                delaunay = self.polylist[poly_index]
                points = delaunay.points

                # Bounding box filtering
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                bbox_mask = np.all(
                    (voxel_coords >= min_coords) & (voxel_coords <= max_coords), axis=1
                )
                candidate_voxel_coords = voxel_coords[bbox_mask]
                candidate_indices_in_voxel_coords = np.where(bbox_mask)[0]

                if candidate_voxel_coords.size == 0:
                    continue  # Skip if no candidate voxels

                # Perform point-in-polyhedron test
                inside = delaunay.find_simplex(candidate_voxel_coords) >= 0
                inside &= ~voxel_checked[candidate_indices_in_voxel_coords]

                if not np.any(inside):
                    continue  # Skip if no inside voxels

                inside_indices_in_voxel_coords = candidate_indices_in_voxel_coords[inside]
                inside_indices = boundary_mask_indices[inside_indices_in_voxel_coords]

                # Compute element indices
                elem_xyz_array = np.ravel_multi_index(inside_indices.T, (Nx, Ny, Nz))

                ridge_pts = self.polyinfo[poly_index]  # Seed indices
                diff = (
                    self.voroTess.voronoi.points[ridge_pts[1]]
                    - self.voroTess.voronoi.points[ridge_pts[0]]
                )
                norm = np.linalg.norm(diff)
                if norm == 0:
                    continue  # Avoid division by zero
                normal = diff / norm

                # Map extended seed indices to original seed indices (crystal labels)
                materials = [self.voroTess.crystal_index_map[pt_idx] for pt_idx in ridge_pts]

                num_voxels = inside_indices.shape[0]
                coords_list.append(inside_indices)
                elem_xyz_list.append(elem_xyz_array)
                materials_array = np.tile(materials, (num_voxels, 1))
                materials_list.append(materials_array)
                normal_array = np.tile(normal, (num_voxels, 1))
                normal_list.append(normal_array)

                # Mark checked voxels
                voxel_checked[inside_indices_in_voxel_coords] = True

            print(
                f"In crystal number-{crystal}, number of marked voxels-{voxel_checked.sum()} out of {len(voxel_checked)} voxels"
            )
            eroded_image[eroded_mask] = crystal

        self.eroded_image = eroded_image

        # Concatenate arrays
        if coords_list:
            self.coords_array = np.vstack(coords_list)
            self.elem_xyz_array = np.concatenate(elem_xyz_list)
            self.materials_array = np.vstack(materials_list)
            self.normal_array = np.vstack(normal_list)
        else:
            self.coords_array = np.empty((0, 3), dtype=int)
            self.elem_xyz_array = np.empty(0, dtype=int)
            self.materials_array = np.empty((0, 2), dtype=int)
            self.normal_array = np.empty((0, 3), dtype=float)

    def _precompute_polyhedrons(self, voroTess: VoronoiTessellation) -> None:
        self.polylist = []
        self.polyinfo = []
        self.polytrack = defaultdict(list)

        for _i, (ridge, ridge_pts) in enumerate(
            zip(
                voroTess.voronoi.ridge_vertices,
                voroTess.voronoi.ridge_points,
                strict=False,
            )
        ):
            if -1 in ridge:
                # Skip infinite ridges
                continue

            points = np.vstack(
                (voroTess.voronoi.vertices[ridge], voroTess.voronoi.points[ridge_pts])
            )
            delaunay = Delaunay(points)
            self.polylist.append(delaunay)
            self.polyinfo.append(ridge_pts)

            # Update polytrack with the correct index
            for pt_idx in ridge_pts:
                crystal_label = self.voroTess.crystal_index_map[pt_idx]
                self.polytrack[crystal_label].append(len(self.polylist) - 1)

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

            # Define dtype for the structured numpy array for GBVoxelInfo
            voxel_info_dtype = np.dtype(
                [
                    ("coords", "i8", (3,)),
                    ("elem_xyz", "i8"),
                    ("materials", "i8", (2,)),
                    ("normal", "f8", (3,)),
                ]
            )
            # Convert lists to a structured numpy array
            voxel_info_array = np.array(
                list(
                    zip(
                        self.coords_array,
                        self.elem_xyz_array,
                        self.materials_array,
                        self.normal_array,
                        strict=False,
                    )
                ),
                dtype=voxel_info_dtype,
            )

            # Create a new field for normals, taking shape from the original image
            Nx, Ny, Nz = self.image.shape
            normals_field = np.zeros((Nx, Ny, Nz, 3))

            for idx, voxel in enumerate(self.coords_array):
                x, y, z = voxel
                normals_field[x, y, z] = self.normal_array[idx]

            # Optionally permute the eroded_image before saving, based on the order parameter
            if order == "xyz":
                #################################
                # Image is in order of x, y, z
                # Vector field is in order of x, y, z
                #################################
                permuted_eroded_image = self.eroded_image
                permuted_voxel_info_array = voxel_info_array.copy()
                permuted_normals_field = normals_field
            elif order == "zyx":
                #################################
                # Image is in order of z, y, x
                # Vector field is in order of x, y, z
                #################################
                permuted_eroded_image = self.eroded_image.transpose(2, 1, 0) 

                permuted_voxel_info_array = voxel_info_array.copy()
                permuted_voxel_info_array["coords"] = voxel_info_array["coords"]
                permuted_voxel_info_array["normal"] = voxel_info_array["normal"]

                permuted_normals_field = normals_field
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
            image_dataset.attrs["interface_thickness"] = self.extrusion_factor*2
            image_dataset.attrs["L"] = self.L
            image_dataset.attrs.create("VoronoiSeeds_xyz", np.array(self.seeds, dtype=np.float64))

            # Save GBVoxelInfo to .h5 file
            if "GBVoxelInfo" in grp:
                del grp["GBVoxelInfo"]
                print("Overwriting existing 'GBVoxelInfo' dataset.")
            grp.create_dataset(
                "GBVoxelInfo",
                data=permuted_voxel_info_array,
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
