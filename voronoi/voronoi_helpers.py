import numpy as np
import h5py

def periodic_difference(pt1, pt2, RVE_length):
    """Compute the periodic difference vector and its norm between two points."""
    diff = pt2 - pt1
    diff = np.where(diff > 0.5 * RVE_length, diff - RVE_length, diff)
    diff = np.where(diff < -0.5 * RVE_length, diff + RVE_length, diff)
    norm_diff = np.linalg.norm(diff)
    return diff, norm_diff

def periodic_dist_matrix(A, B, RVE_length):
        """Compute periodic distance matrix between points in A and B."""
        # Expand dimensions
        A = A[:, np.newaxis, :]
        B = B[np.newaxis, :, :]

        # Compute the periodic difference between all combinations of points in A and B
        diff = A - B
        diff = np.where(diff > 0.5 * RVE_length, diff - RVE_length, diff)
        diff = np.where(diff < -0.5 * RVE_length, diff + RVE_length, diff)

        # Compute the squared distances
        dist_sq = np.sum(diff**2, axis=-1)
        return dist_sq

from scipy.ndimage import binary_erosion
def periodic_erosion(mask, shrink_factor):
    """Applies erosion on a binary mask while considering periodic boundaries."""
    selem = np.ones((shrink_factor, shrink_factor, shrink_factor))
    padded_mask = np.pad(mask, ((shrink_factor//2, shrink_factor//2), 
                                (shrink_factor//2, shrink_factor//2), 
                                (shrink_factor//2, shrink_factor//2)), mode='wrap')
    eroded_padded_mask = binary_erosion(padded_mask, structure=selem)
    Nx, Ny, Nz = mask.shape
    return eroded_padded_mask[shrink_factor//2:Nx+shrink_factor//2, 
                              shrink_factor//2:Ny+shrink_factor//2, 
                              shrink_factor//2:Nz+shrink_factor//2]

def save_sample_to_hdf5(dsetname_prefix, tessellation, filename, seeds=None, lattice_vectors=None, neighbors=None, GBVoxelInfo_list=[]):
    """
    Save a single 3D Voronoi tessellation sample to an HDF5 file.
    
    Parameters:
    - dsetname_prefix: Prefix of the dataset names. The datasets will be named with 
      the prefix followed by '/image', '/neighbors', '/seed_positions',  '/lattice_vectors', and '/GBVoxelInfo'.
    - tessellation: The 3D Voronoi tessellation to save.
    - filename: Name of the HDF5 file to which the data should be saved.
    - seeds: Optional. The seed points corresponding to the tessellation.
    - lattice_vectors: Optional. The lattice vectors for each crystal in the tessellation.
    - neighbors: Optional. Dictionary containing neighbor relationships for each crystal.
    - GBVoxelInfo_list: Optional. List of GBVoxelInfo objects detailing grain boundary information.
    """
    
    with h5py.File(filename, 'a') as f:
        # Create a group with the dsetname_prefix
        grp = f.create_group(dsetname_prefix)
        compression_opts = 9
        
        # Always save tessellation
        grp.create_dataset('image', data=tessellation, dtype=np.int32, compression="gzip", compression_opts=compression_opts)
        
        # Save seeds if provided
        if seeds is not None:
            grp.create_dataset('seed_positions', data=seeds, dtype=np.int32, compression="gzip", compression_opts=compression_opts)

        # Save lattice vectors if provided
        if lattice_vectors is not None:
            grp.create_dataset('lattice_vectors', data=lattice_vectors, dtype=np.float64, compression="gzip", compression_opts=compression_opts)    
        
        # Save neighbors if provided
        if neighbors is not None:
            # Determine n_max_neighbors from the neighbors dictionary
            n_max_neighbors = max(len(neigh_list) for neigh_list in neighbors.values())
            
            # Create a padded 2D array for neighbor data
            neighbor_array = -1 * np.ones((len(seeds), n_max_neighbors), dtype=np.int32)  # -1 as the padding value
            for idx, neigh_list in neighbors.items():
                neighbor_array[idx, :len(neigh_list)] = neigh_list
            
            grp.create_dataset('neighbors', data=neighbor_array, dtype=np.int32, compression="gzip", compression_opts=compression_opts)
        
        # If GBVoxelInfo_list is provided and not empty
        if GBVoxelInfo_list:
            # Define dtype for the structured numpy array
            voxel_info_dtype = np.dtype([
                ('coords', 'i8', (3,)),
                ('elem_xyz', 'i8'),
                ('materials', 'i8', (2,)),
                ('normal', 'f8', (3,))
            ])

            # Convert GBVoxelInfo list to structured numpy array
            voxel_info_array = np.array([(voxel.coords, voxel.elem_xyz, voxel.materials, voxel.normal) 
                                         for voxel in GBVoxelInfo_list], dtype=voxel_info_dtype)
            
            # Save structured numpy array to .h5 file
            grp.create_dataset('GBVoxelInfo', data=voxel_info_array, dtype=voxel_info_dtype, compression="gzip", compression_opts=compression_opts)

            # Create a new field for normal
            Nx, Ny, Nz = tessellation.shape
            normals_field = np.zeros((Nx, Ny, Nz, 3))

            for voxel in GBVoxelInfo_list:
                x, y, z = voxel.coords
                normals_field[x, y, z] = voxel.normal

            grp.create_dataset('normal', data=normals_field, dtype='f8', compression="gzip", compression_opts=compression_opts)

def load_data_from_h5(filename, dsetname_prefix):
    """
    Load data (image, neighbors, seeds) from an HDF5 file based on a dataset name prefix.
    
    Parameters:
    - filename: Path to the HDF5 file.
    - dsetname_prefix: Prefix of the dataset names. Assumes the datasets are named with 
      the prefix followed by '/image', '/neighbors', and '/seed_positions'.
    
    """

    with h5py.File(filename, 'r') as f:
        # Try to load the image dataset
        try:
            image = f[dsetname_prefix + '/image'][:]
        except KeyError:
            try:
                image = f[dsetname_prefix + '/ms'][:]
            except KeyError:
                print("Error: Neither /image nor /ms datasets found.")
                image = None
        
        # Try to load neighbors
        try:
            neighbors_padded = f[dsetname_prefix + '/neighbors'][:]
            # Assuming padding value is -1 for neighbors, remove it
            padding_value = -1
            neighbors_list = [list(filter(lambda x: x != padding_value, row)) for row in neighbors_padded]
            # Convert the list of neighbors into a dictionary
            neighbors = {i: neigh for i, neigh in enumerate(neighbors_list)}
        except KeyError:
            print("Warning: Neighbors dataset not found.")
            neighbors = None
        
        # Try to load seeds
        try:
            seeds = f[dsetname_prefix + '/seed_positions'][:]
        except KeyError:
            print("Warning: seed_positions dataset not found.")
            seeds = None

    return image, seeds, neighbors



def calculate_polygon_area_3d(vertices, normal):
    """
    Calculate the area of a polygon in 3D.

    Parameters:
    - vertices: List of 3D coordinates representing the vertices of the polygon.
    - normal: 3D vector representing the normal of the polygon.

    Returns:
    - area: The area of the polygon.

    This function calculates the area of a polygon in 3D by projecting the vertices onto a plane spanned by two orthogonal vectors and then using the Shoelace formula to calculate the area in 2D.
    """
    # Ensure normal is a unit vector
    normal = normal / np.linalg.norm(normal)

    # Find a vector that is not parallel to the normal
    if not np.isclose(normal[0], 0) or not np.isclose(normal[1], 0):
        non_parallel_vector = np.array([0, 0, 1])
    else:
        non_parallel_vector = np.array([1, 0, 0])

    # Find two vectors that are orthogonal to the normal
    u = np.cross(normal, non_parallel_vector)
    v = np.cross(normal, u)

    # Normalize u and v to form an orthonormal basis with the normal
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # Project vertices onto the plane spanned by u and v
    vertices_2d = np.array([[np.dot(vertex - vertices[0], u), np.dot(vertex - vertices[0], v)] for vertex in vertices])

    # Use the Shoelace formula to calculate the area of the polygon in 2D
    x, y = vertices_2d[:, 0], vertices_2d[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

















































































# import matplotlib.pyplot as plt
# def visualize_voronoi_3d(cube, seeds=None):
#     """
#     Visualize a 3D Voronoi tessellation using matplotlib's voxels.
    
#     Parameters:
#     - cube: 3D array containing the Voronoi tessellation.
#     - seeds: Optional. Original seed points to be plotted over the tessellation.
#     """
    
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # Define colors for each Voronoi region
#     unique_vals = np.unique(cube)
#     colors = plt.cm.jet(np.linspace(0, 1, len(unique_vals)))
#     color_map = dict(zip(unique_vals, colors))
#     colored_cube = np.array([color_map[val] for val in cube.ravel()])
#     colored_cube = colored_cube.reshape(*cube.shape, 4)

#     mask = cube >= 0  # Use this mask to fill voxels
#     ax.voxels(mask, facecolors=colored_cube, edgecolor='k', linewidth=0.7)
#     ax.set_box_aspect([1, 1, 1])

#     if seeds is not None:
#         ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], c='red', marker='x', s=100)
    
#     plt.title("3D Voronoi Tessellation")
#     plt.show()