import numpy as np
import os
import subprocess
from voronoi_helpers import *

if __name__ == '__main__':
    # Define parameters for the Voronoi tessellation
    Nx, Ny, Nz = 128, 128, 128
    num_crystals = 64
    h5filename = 'voronoi_data_3d.h5'
    RVE_length = [1, 1, 1]

    # Check if the file already exists and, if so, delete it
    if os.path.exists(h5filename):
        os.remove(h5filename)

    seeds, lattice_vectors = generate_seeds(num_crystals, RVE_length, "random")
    neighbors, _ = find_neighbors(seeds, RVE_length)
    tessellation, seeds = generate_periodic_voronoi(Nx, Ny, Nz, num_crystals, RVE_length, seeds)
    save_sample_to_hdf5("/dset_0", tessellation=tessellation, filename=h5filename, seeds=seeds, neighbors=neighbors)    

    tessellation, seeds = generate_periodic_voronoi(Nx*4, Ny*4, Nz*4, num_crystals, RVE_length, seeds)
    save_sample_to_hdf5("/dset_1", tessellation=tessellation, filename=h5filename, seeds=seeds, neighbors=neighbors)
    
    # # Erode the loaded tessellation.
    # shrink_factor = 3
    # shrunk, GBVoxelInfo = shrink_crystals_periodically(tessellation, seeds, neighbors, shrink_factor, RVE_length)
    # save_sample_to_hdf5("/dset_0/eroded_"+str(shrink_factor), tessellation=shrunk, filename=h5filename, seeds=seeds, neighbors=neighbors, GBVoxelInfo_list=GBVoxelInfo)    

    subprocess.run(["python", "h52xdmf.py", "-v", h5filename], check=True)
























# crystal_index = 32
#     mask = np.isin(tessellation, [crystal_index] + neighbors[crystal_index])
#     tessellation = np.where(mask, tessellation, 0)
#     save_sample_to_hdf5("/dset_1", tessellation, h5filename, seeds, neighbors)
