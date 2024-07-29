import numpy as np
import h5py

class GBVoxelInfo:
    def __init__(self, coords, elem_xyz, materials, normal):
        self.coords = coords  # 3D coordinates of the grain boundary voxel
        self.elem_xyz = elem_xyz  # Element numbers when indexed with loops (x, y, z)
        self.materials = materials  # Crystals involved at this grain boundary
        self.normal = normal  # Normal vector of the boundary

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

from scipy.ndimage import binary_erosion, sobel
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

def shrink_crystals_periodically(cube, seeds, neighbors, shrink_factor=5, RVE_length=[1,1,1]):
    """Erodes (shrinks) each unique crystal in the provided 3D cube image."""
    
    RVE_length = np.array(RVE_length)
    unique_crystals = np.unique(cube)
    eroded_cube = -1 * np.ones_like(cube)
    
    # A helper function to get the edges using Sobel operator
    def get_edges(binary_image):
        edges = np.sqrt(sobel(binary_image, axis=0)**2 + sobel(binary_image, axis=1)**2 + sobel(binary_image, axis=2)**2)
        return edges > 0
    
    boundary_masks = {}
    crystal_edges = {}
    for crystal in unique_crystals:
        crystal_mask = (cube == crystal)
        eroded_mask = periodic_erosion(crystal_mask, shrink_factor)
        boundary_masks[crystal] = np.logical_and(crystal_mask, np.logical_not(eroded_mask))
        eroded_cube[eroded_mask] = crystal
        # crystal_edges[crystal] = get_edges(eroded_mask)

    grain_boundary_info = []
    [Nx, Ny, Nz] = cube.shape
    [hx, hy, hz] = RVE_length/np.array([Nx, Ny, Nz])

    # edge_voxels_dict = {crystal: np.array(np.where(crystal_edges[crystal])).T * np.array([hx, hy, hz])
    #                 for crystal in unique_crystals}
    
    # for crystal in unique_crystals:
    #     boundary_voxels = np.array(np.where(boundary_masks[crystal])).T * np.array([hx, hy, hz])
        
    #     min_distances = np.full(boundary_voxels.shape[0], np.inf)
    #     closest_crystals = np.full(boundary_voxels.shape[0], -1)

    #     for neighbor in neighbors[crystal]:
    #         edge_voxels = edge_voxels_dict[neighbor]
            
    #         # Using broadcasting for distance computation
    #         dist_matrix = periodic_dist_matrix(boundary_voxels, edge_voxels, RVE_length)
    #         min_neighbor_distances = np.min(dist_matrix, axis=1)
            
    #         mask = min_neighbor_distances < min_distances
    #         min_distances[mask] = min_neighbor_distances[mask]
    #         closest_crystals[mask] = neighbor

    #     for idx, voxel in enumerate(boundary_voxels):
    #         materials = sorted([crystal, closest_crystals[idx]])
    #         diff, norm_diff = periodic_difference(seeds[materials[0]], seeds[materials[1]], RVE_length)
    #         normal = diff[::-1] / norm_diff
    #         x, y, z = voxel
    #         elem_xyz = (z % Nz) + (y % Ny) * Nz + (x % Nx) * Ny * Nz
    #         voxel_info = GBVoxelInfo((int(x/hx), int(y/hy), int(z/hz)), elem_xyz, materials, normal)
    #         grain_boundary_info.append(voxel_info)

    return eroded_cube, grain_boundary_info

from scipy.spatial import Delaunay
def find_neighbors(seeds, RVE_length):
    """
    Find neighboring crystals (seed points) for each crystal using Delaunay triangulation, 
    taking periodicity into account.

    Parameters:
    - seeds: The seed points corresponding to the tessellation.
    - RVE_length: Lengths of the representative volume element along x, y, and z.
    
    Returns:
    - neighbors: Dictionary where keys are the indices of the seed points, and 
                 values are lists of neighboring seed indices.
    """
    
    # Create replicated points across the domain for periodic boundary conditions
    offsets = np.array([
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
        [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],

        [0, -1, -1], [0, -1, 0], [0, -1, 1],
        [0, 0, -1],                [0, 0, 1],
        [0, 1, -1], [0, 1, 0], [0, 1, 1],

        [1, -1, -1], [1, -1, 0], [1, -1, 1],
        [1, 0, -1], [1, 0, 0], [1, 0, 1],
        [1, 1, -1], [1, 1, 0], [1, 1, 1],
    ])

    replicated_seeds = [seeds + offset * RVE_length for offset in offsets]
    all_seeds = np.vstack([seeds] + replicated_seeds)
    
    # Apply Delaunay triangulation
    tri = Delaunay(all_seeds)
    
    # Extract neighboring relationships
    neighbors = {}
    for simplex in tri.simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    if i < len(seeds):  # Ensure we're looking only at original seeds, not replicas
                        if i not in neighbors:
                            neighbors[i] = set()
                        # Map replicas back to their original counterparts
                        # neighbors[i].add(j % len(seeds))
                        mapped_j = j % len(seeds)
                        if i != mapped_j:
                            neighbors[i].add(mapped_j)
    
    # Convert sets to lists
    for key in neighbors:
        neighbors[key] = sorted(list(neighbors[key]))

    return neighbors, tri

from scipy.spatial import cKDTree as KDTree
def generate_periodic_voronoi(Nx, Ny, Nz, num_crystals, RVE_length=[1, 1, 1], provided_seeds=None):
    """
    Generate a periodic 3D Voronoi tessellation using periodic boundary conditions.
    
    Parameters:
    - Nx, Ny, Nz: Dimensions of the 3D grid.
    - num_crystals: Number of seed points (crystals) to generate within the grid.
    - RVE_length: Lengths of the representative volume element along x, y, and z.
    - provided_seeds: Optional. An array of seed points in the range [0,0,0] to RVE_length. If provided and compatible, these will be used.
    
    Returns:
    - cube: A 3D array of size Nx x Ny x Nz containing Voronoi labels.
    - seeds: The original seed points within the specified range.
    """
    
    # Check if provided seeds are compatible
    if provided_seeds is not None and len(provided_seeds) == num_crystals:
        seeds = provided_seeds
    else:
        # Generate random seeds within the domain if not provided or if provided seeds are not compatible
        seeds = generate_seeds(Nx, Ny, Nz, num_crystals, RVE_length=[1, 1, 1], method="random")


    dx, dy, dz = RVE_length[0] / Nx, RVE_length[1] / Ny, RVE_length[2] / Nz
    # Create a mesh grid of points within the main domain, and adjust for voxel centers
    x, y, z = np.meshgrid(np.linspace(0.5*dx, RVE_length[0] - 0.5*dx, Nx),
                          np.linspace(0.5*dy, RVE_length[1] - 0.5*dy, Ny),
                          np.linspace(0.5*dz, RVE_length[2] - 0.5*dz, Nz),
                          indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Use KDTree with periodic boundary conditions to find the nearest seed point for each point in the domain
    tree = KDTree(seeds, boxsize=RVE_length)
    _, labels = tree.query(points)
    cube = labels.reshape(Nx, Ny, Nz)

    return cube, seeds

def generate_seeds(Nx, Ny, Nz, num_crystals, RVE_length=[1, 1, 1], method="random"):
    """
    Generate seed points (crystals) for 3D Voronoi tessellations and their associated orthonormal lattice vectors.
    
    Parameters:
    - Nx, Ny, Nz: Dimensions of the volume in which seeds should be generated.
    - num_crystals: Number of seed points (or crystals) to generate.
    - RVE_length: Dimensions of the representative volume element (RVE).
    - method: Method to generate seed points. Can be "random".
    
    Returns:
    - seeds: Generated seed points of shape (num_crystals, 3).
    - lattice_vectors: Orthonormal lattice vectors for each seed. The array has a shape 
      of (num_crystals, 3, 3), where:
        * The first dimension corresponds to each crystal.
        * The second dimension enumerates the three orthonormal vectors for each crystal.
        * The third dimension represents the three components of a vector (x, y, z).

    Notes:
    A uniform distribution on the group of rotations SO(3) is used to assign random orientations
    to the grains of the aggregate. This is based on random variables X, Y, Z that are uniformly
    distributed on the interval [0, 1) to determine three Euler angles (z-x-z convention) via 
    ϕ1 = 2πX, Φ = acos(2Y - 1), ϕ2 = 2πZ.
    """

    # Generate seeds based on method
    if method == "random":
        seeds = np.random.rand(num_crystals, 3) * np.array(RVE_length)
    
    else:
        raise ValueError("Unknown sampling method!")
    
    # Generate random orientations using the Euler angles (z-x-z convention)
    X, Y, Z = np.random.rand(3, num_crystals)
    phi1 = 2 * np.pi * X
    Phi = np.arccos(2 * Y - 1)
    phi2 = 2 * np.pi * Z

    # Convert Euler angles to rotation matrices
    c1, s1 = np.cos(phi1), np.sin(phi1)
    cP, sP = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    
    R11 = c1*c2 - cP*s1*s2
    R12 = -c1*s2 - cP*c2*s1
    R13 = s1*sP
    R21 = c2*s1 + c1*cP*s2
    R22 = c1*cP*c2 - s1*s2
    R23 = -c1*sP
    R31 = sP*s2
    R32 = c2*sP
    R33 = cP
    
    rot_matrices = np.stack((R11, R12, R13, R21, R22, R23, R31, R32, R33), axis=-1).reshape(num_crystals, 3, 3)
    
    # The initial orientation is simply an identity matrix, representing [1, 0, 0], [0, 1, 0], and [0, 0, 1]
    # Using the rotation matrices, we compute the lattice vectors for each seed
    initial_orientation = np.eye(3)
    lattice_vectors = np.einsum('nij,jk->nik', rot_matrices, initial_orientation)

    return seeds, lattice_vectors

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

import matplotlib.pyplot as plt
def visualize_voronoi_3d(cube, seeds=None):
    """
    Visualize a 3D Voronoi tessellation using matplotlib's voxels.
    
    Parameters:
    - cube: 3D array containing the Voronoi tessellation.
    - seeds: Optional. Original seed points to be plotted over the tessellation.
    """
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each Voronoi region
    unique_vals = np.unique(cube)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_vals)))
    color_map = dict(zip(unique_vals, colors))
    colored_cube = np.array([color_map[val] for val in cube.ravel()])
    colored_cube = colored_cube.reshape(*cube.shape, 4)

    mask = cube >= 0  # Use this mask to fill voxels
    ax.voxels(mask, facecolors=colored_cube, edgecolor='k', linewidth=0.7)
    ax.set_box_aspect([1, 1, 1])

    if seeds is not None:
        ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], c='red', marker='x', s=100)
    
    plt.title("3D Voronoi Tessellation")
    plt.show()









































