import numpy as np
from voronoi_helpers import periodic_difference, periodic_dist_matrix, periodic_erosion
import h5py
from scipy.spatial import Delaunay
import logging

class PeriodicVoronoiImageErosion:
    def __init__(self, voroImg, voroTess, shrink_factor=5):
        self.image = voroImg.image
        self.seeds = voroTess.seeds
        self.shrink_factor = shrink_factor
        self.RVE_length = np.array(voroImg.RVE_length)
        self.eroded_image = None
        self.N = np.array(voroImg.N)
        self.voroTess = voroTess

        # Initialize lists to store grain boundary information
        self.coords_list = []
        self.elem_xyz_list = []
        self.materials_list = []
        self.normal_list = []

        self._precompute_polyhedrons(voroTess)
        self._shrink_and_analyze()

    def _shrink_and_analyze(self):
        unique_crystals = np.unique(self.image)
        eroded_image = -1 * np.ones_like(self.image)
        Nx, Ny, Nz = self.image.shape
        hx, hy, hz = self.RVE_length / np.array([Nx, Ny, Nz])

        # Function to perform batched Delaunay check
        def batch_check_delaunay(voxel_coords, delaunay):
            return delaunay.find_simplex(voxel_coords) >= 0

        for crystal in unique_crystals:
            crystal_mask = (self.image == crystal)
            eroded_mask = periodic_erosion(crystal_mask, self.shrink_factor)
            boundary_mask = np.logical_and(crystal_mask, np.logical_not(eroded_mask))
            boundary_mask_indices = np.array(np.where(boundary_mask)).T  # Indices of boundary voxels
            voxel_coords = (boundary_mask_indices + 0.5) * np.array([hx, hy, hz])  # Convert indices to physical coordinates

            # Initialize a boolean array to keep track of checked voxels
            voxel_checked = np.full(len(voxel_coords), False, dtype=bool)

            # for poly_index in range(0,len(self.polylist)): #self.polytrack[crystal]:  # Only check polyhedrons associated with this crystal
            for poly_index in self.polytrack[crystal]:  # Only check polyhedrons associated with this crystal
                delaunay = self.polylist[poly_index]
                inside = batch_check_delaunay(voxel_coords, delaunay)
                inside_indices = boundary_mask_indices[inside & ~voxel_checked]  # Filter voxels that are inside and not checked yet

                ridge_pts = self.polyinfo[poly_index]  # Get the seeds that define this polyhedron
                materials = [self.voroTess.crystal_index_map[pt_idx] for pt_idx in ridge_pts]
                diff = self.voroTess.voronoi.points[ridge_pts[1]] - self.voroTess.voronoi.points[ridge_pts[0]]
                normal = diff / np.linalg.norm(diff)  # Normalized vector joining the seeds

                for voxel_idx in inside_indices:
                    elem_xyz = np.ravel_multi_index(voxel_idx, (Nx, Ny, Nz))
                    self.coords_list.append(voxel_idx)
                    self.elem_xyz_list.append(elem_xyz)
                    self.materials_list.append(materials)
                    self.normal_list.append(normal)
                
                voxel_checked[inside] = True  # Mark checked voxels
            
            print(f"In crystal number-{crystal}, number of marked voxels-{sum(voxel_checked)} out of {len(voxel_checked)} voxels")
            eroded_image[eroded_mask] = crystal

        self.eroded_image = eroded_image
        # Convert lists to NumPy arrays for further processing if needed
        self.coords_array = np.array(self.coords_list)
        self.elem_xyz_array = np.array(self.elem_xyz_list)
        self.materials_array = np.array(self.materials_list)
        self.normal_array = np.array(self.normal_list)




    def write_to_h5(self, dsetname_prefix, filename, order="xyz"):
        with h5py.File(filename, 'a') as h5_file:
            grp = h5_file.require_group(dsetname_prefix)
            compression_opts = 9

            # Define dtype for the structured numpy array for GBVoxelInfo
            voxel_info_dtype = np.dtype([
                ('coords', 'i8', (3,)),
                ('elem_xyz', 'i8'),
                ('materials', 'i8', (2,)),
                ('normal', 'f8', (3,))
            ])
            # Convert lists to a structured numpy array
            voxel_info_array = np.array(list(zip(self.coords_list, self.elem_xyz_list, self.materials_list, self.normal_list)), dtype=voxel_info_dtype)

            # Create a new field for normals, taking shape from the original image
            Nx, Ny, Nz = self.image.shape
            normals_field = np.zeros((Nx, Ny, Nz, 3))

            for idx, voxel in enumerate(self.coords_list):
                x, y, z = voxel
                normals_field[x, y, z] = self.normal_list[idx]

            # Optionally permute the eroded_image before saving, based on the order parameter
            if order == "xyz":
                permuted_eroded_image = self.eroded_image
                permuted_voxel_info_array = voxel_info_array.copy()
                permuted_normals_field = normals_field
            elif order == "zyx":
                permuted_eroded_image = self.eroded_image.transpose(2, 1, 0)  # Permute spatial dimensions

                permuted_voxel_info_array = voxel_info_array.copy()
                permuted_voxel_info_array['coords'] = voxel_info_array['coords'][:, ::-1]
                permuted_voxel_info_array['normal'] = voxel_info_array['normal'][:, ::-1]

                permuted_normals_field = normals_field
                permuted_normals_field = normals_field.transpose(2, 1, 0, 3)  # Permute spatial dimensions
                #permuted_normals_field = permuted_normals_field[..., [2, 1, 0]]  # Reverse the vector components

            # Save eroded image to .h5 file
            grp.create_dataset('eroded_image', data=permuted_eroded_image, dtype=np.int32, compression="gzip", compression_opts=compression_opts)
            # Save GBVoxelInfo to .h5 file
            grp.create_dataset('GBVoxelInfo', data=permuted_voxel_info_array, compression="gzip", compression_opts=compression_opts)
            # Save normals to .h5 file
            grp.create_dataset('normals', data=permuted_normals_field, dtype='f8', compression="gzip", compression_opts=compression_opts)


    def _precompute_polyhedrons(self, voroTess):
        # Initialize the list for all polyhedrons' Delaunay triangulations
        self.polylist = []

        # Initialize the array to store information about which two seeds each polyhedron belongs to
        self.polyinfo = np.zeros((len(voroTess.voronoi.ridge_vertices), 2), dtype=int)

        # Initialize the list to track which polyhedrons belong to which seed
        self.polytrack = [[] for _ in range(len(voroTess.seeds))]

        for i, (ridge, ridge_pts) in enumerate(zip(voroTess.voronoi.ridge_vertices, voroTess.voronoi.ridge_points)):
            # Combine ridge vertices with seed points to form a potential polyhedron
            points = np.vstack((voroTess.voronoi.vertices[ridge], voroTess.voronoi.points[ridge_pts]))

            # Compute the Delaunay triangulation of these points
            delaunay = Delaunay(points)

            # Add the Delaunay triangulation to the polylist
            self.polylist.append(delaunay)

            # Store which two seeds the polyhedron belongs to in polyinfo
            self.polyinfo[i] = sorted(ridge_pts)

            # Update the polytrack to associate this polyhedron with its seeds
            for pt_idx in ridge_pts:
                self.polytrack[voroTess.crystal_index_map[pt_idx]].append(i)



    # def _precompute_polyhedrons(self, voroTess):
    #     # Initialize the list for all polyhedrons' Delaunay triangulations
    #     self.polylist = []

    #     # Initialize the array to store information about which two seeds each polyhedron belongs to
    #     self.polyinfo = np.zeros((len(voroTess.orig_ridges), 2), dtype=int)

    #     # Initialize the list to track which polyhedrons belong to which seed
    #     self.polytrack = [[] for _ in range(len(voroTess.seeds))]

    #     for i, (ridge, ridge_pts) in enumerate(zip(voroTess.orig_ridges, voroTess.orig_ridge_pts)):
    #         # Combine ridge vertices with seed points to form a potential polyhedron
    #         points = np.vstack((voroTess.voronoi.vertices[ridge], voroTess.voronoi.points[ridge_pts]))

    #         # Compute the Delaunay triangulation of these points
    #         delaunay = Delaunay(points)

    #         # Add the Delaunay triangulation to the polylist
    #         self.polylist.append(delaunay)

    #         # Store which two seeds the polyhedron belongs to in polyinfo
    #         self.polyinfo[i] = ridge_pts

    #         # Update the polytrack to associate this polyhedron with its seeds
    #         for pt_idx in ridge_pts:
    #             self.polytrack[voroTess.crystal_index_map[pt_idx]].append(i)



































    



    # def _shrink_and_analyze(self):
    #         unique_crystals = np.unique(self.image)
    #         eroded_image = -1 * np.ones_like(self.image)
    #         Nx, Ny, Nz = self.image.shape
    #         hx, hy, hz = self.RVE_length / np.array([Nx, Ny, Nz])

    #         for crystal in unique_crystals:
    #             crystal_mask = (self.image == crystal)
    #             eroded_mask = periodic_erosion(crystal_mask, self.shrink_factor)
    #             boundary_mask = np.logical_and(crystal_mask, np.logical_not(eroded_mask))  # Corrected boundary mask definition
    #             boundary_mask_indices = np.array(np.where(boundary_mask)).T  # Indices of boundary voxels

    #             for voxel_idx in boundary_mask_indices:
    #                 voxel_coords = (voxel_idx + 0.5) * np.array([hx, hy, hz])  # Convert indices to physical coordinates

    #                 # Check which polyhedron this voxel belongs to
    #                 # for poly_index in self.polytrack[crystal]:  # Only check polyhedrons associated with this crystal
    #                 for poly_index in range(0,len(self.polylist)):  # Only check polyhedrons associated with this crystal
    #                     delaunay = self.polylist[poly_index]
    #                     if delaunay.find_simplex(voxel_coords) >= 0:  # If voxel is inside the Delaunay triangulation
    #                         ridge_pts = self.polyinfo[poly_index]  # Get the seeds that define this polyhedron
    #                         materials = sorted([self.voroTess.crystal_index_map[pt_idx] for pt_idx in ridge_pts])
    #                         diff = self.voroTess.voronoi.points[ridge_pts[1]] - self.voroTess.voronoi.points[ridge_pts[0]]
    #                         normal = diff / np.linalg.norm(diff)  # Normalized vector joining the seeds

    #                         # Convert voxel indices to linear index
    #                         elem_xyz = np.ravel_multi_index(voxel_idx, (Nx, Ny, Nz))

    #                         # Append information to lists
    #                         self.coords_list.append(voxel_idx)  # Use voxel indices for consistency
    #                         self.elem_xyz_list.append(elem_xyz)
    #                         self.materials_list.append(materials)
    #                         self.normal_list.append(normal)
    #                         break  # Stop checking other polyhedrons once the correct one is found
                
    #             # print(f"In crystal number-{crystal}, number of marked voxels-{sum(voxel_checked)} out of {len(voxel_checked)} voxels")
    #             eroded_image[eroded_mask] = crystal  # Update eroded image

    #         self.eroded_image = eroded_image
    #         # Convert lists to NumPy arrays for further processing if needed
    #         self.coords_array = np.array(self.coords_list)
    #         self.elem_xyz_array = np.array(self.elem_xyz_list)
    #         self.materials_array = np.array(self.materials_list)
    #         self.normal_array = np.array(self.normal_list)















# def _shrink_and_analyze(self):
#         unique_crystals = np.unique(self.image)
#         eroded_image = -1 * np.ones_like(self.image)
#         Nx, Ny, Nz = self.image.shape
#         hx, hy, hz = self.RVE_length / np.array([Nx, Ny, Nz])

#         for crystal in unique_crystals:
#             crystal_mask = (self.image == crystal)
#             eroded_mask = periodic_erosion(crystal_mask, self.shrink_factor)
#             boundary_mask = np.logical_and(crystal_mask, np.logical_not(eroded_mask))
            
#             boundary_voxels = (np.array(np.where(boundary_mask)).T + 0.5) * np.array([hx, hy, hz])
#             neighboring_seeds = self.seeds[self.neighbors[crystal]]

#             d_AB = periodic_dist_matrix(boundary_voxels, neighboring_seeds, self.RVE_length)
#             closest_neighbor_indices = np.argmin(d_AB, axis=1)

#             for idx, voxel in enumerate(np.array(np.where(boundary_mask)).T):
#                 closest_neighbor = self.neighbors[crystal][closest_neighbor_indices[idx]]
#                 materials = sorted([crystal, closest_neighbor])
#                 diff, norm_diff = periodic_difference(self.seeds[materials[0]], self.seeds[materials[1]], self.RVE_length)
#                 normal = diff[::-1] / norm_diff
#                 elem_xyz = voxel[2] % Nz + (voxel[1] % Ny) * Nz + (voxel[0] % Nx) * Ny * Nz
                
#                 # Append information to lists
#                 self.coords_list.append(voxel) 
#                 self.elem_xyz_list.append(elem_xyz)
#                 self.materials_list.append(materials)
#                 self.normal_list.append(normal)

#             eroded_image[eroded_mask] = crystal

#         self.eroded_image = eroded_image
#         # Convert lists to NumPy arrays
#         self.coords_array = np.array(self.coords_list)
#         self.elem_xyz_array = np.array(self.elem_xyz_list)
#         self.materials_array = np.array(self.materials_list)
#         self.normal_array = np.array(self.normal_list)









