import numpy as np
from voronoi_helpers import periodic_erosion
import h5py
from scipy.spatial import Delaunay
from collections import defaultdict  # Import defaultdict for convenient dictionary handling

class PeriodicVoronoiImageErosion:
    def __init__(self, voroImg, voroTess, shrink_factor=5):
        self.image = voroImg.image
        self.seeds = voroTess.seeds
        self.shrink_factor = shrink_factor
        self.RVE_length = np.array(voroImg.L)
        self.eroded_image = None
        self.N = np.array(voroImg.resolution)
        self.voroTess = voroTess

        # Initialize lists to store grain boundary information
        self.coords_list = []
        self.elem_xyz_list = []
        self.materials_list = []
        self.normal_list = []

        self._precompute_polyhedrons(voroTess)
        self._shrink_and_analyze()

    def _minimum_image_distance(self, pos1, pos2):
        delta = pos1 - pos2
        delta -= self.RVE_length * np.round(delta / self.RVE_length)
        return delta

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

            # Use the crystal label as a key in the dictionary
            for poly_index in self.polytrack.get(crystal, []):
                delaunay = self.polylist[poly_index]
                inside = batch_check_delaunay(voxel_coords, delaunay)
                inside_indices = boundary_mask_indices[inside & ~voxel_checked]  # Filter voxels that are inside and not checked yet

                ridge_pts = self.polyinfo[poly_index]  # Get the seeds that define this polyhedron
                materials = [self.voroTess.crystal_index_map[pt_idx] for pt_idx in ridge_pts]

                # Compute diff considering periodicity
                diff = self._minimum_image_distance(
                    self.voroTess.voronoi.points[ridge_pts[1]],
                    self.voroTess.voronoi.points[ridge_pts[0]]
                )
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

    def _precompute_polyhedrons(self, voroTess):
        self.polylist = []
        self.polyinfo = []
        self.polytrack = defaultdict(list)

        for i, (ridge, ridge_pts) in enumerate(zip(voroTess.voronoi.ridge_vertices, voroTess.voronoi.ridge_points)):
            if -1 in ridge:
                # Skip infinite ridges
                continue

            points = np.vstack((voroTess.voronoi.vertices[ridge], voroTess.voronoi.points[ridge_pts]))
            delaunay = Delaunay(points)
            self.polylist.append(delaunay)
            self.polyinfo.append(ridge_pts)

            # Update polytrack with the correct index
            for pt_idx in ridge_pts:
                crystal_label = self.voroTess.crystal_index_map[pt_idx]
                self.polytrack[crystal_label].append(len(self.polylist) - 1)
                

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
            if 'eroded_image' in grp:
                del grp['eroded_image']
                print("Overwriting existing 'eroded_image' dataset.")
            grp.create_dataset('eroded_image', data=permuted_eroded_image, dtype=np.int32, compression="gzip", compression_opts=compression_opts)

            # Save GBVoxelInfo to .h5 file
            if 'GBVoxelInfo' in grp:
                del grp['GBVoxelInfo']
                print("Overwriting existing 'GBVoxelInfo' dataset.")
            grp.create_dataset('GBVoxelInfo', data=permuted_voxel_info_array, compression="gzip", compression_opts=compression_opts)

            # Save normals to .h5 file
            if 'normals' in grp:
                del grp['normals']
                print("Overwriting existing 'normals' dataset.")
            grp.create_dataset('normals', data=permuted_normals_field, dtype='f8', compression="gzip", compression_opts=compression_opts)
            
            