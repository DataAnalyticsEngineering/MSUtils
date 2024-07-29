#!/opt/homebrew/Caskroom/miniforge/base/envs/tensorflow/bin/python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

def generate_periodic_voronoi_2d(Nx, Ny, num_crystals):
    """
    Generate a periodic 2D Voronoi tessellation.
    
    Parameters:
    - Nx, Ny: Dimensions of the 2D grid.
    - num_crystals: Number of seed points (crystals) to generate within the grid.
    
    Returns:
    - image: A 2D array of size Nx x Ny containing Voronoi labels.
    - seeds: The original seed points within the main grid.
    """
    
    seeds = np.random.rand(num_crystals, 2) * np.array([Nx, Ny])
    replications = [-1, 0, 1]
    all_seeds = []
    for dx in replications:
        for dy in replications:
            offset = np.array([dx*Nx, dy*Ny])
            all_seeds.append(seeds + offset)
    all_seeds = np.vstack(all_seeds)

    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel()))

    distances = cdist(points, all_seeds)
    labels = np.argmin(distances, axis=1) % num_crystals
    image = labels.reshape(Nx, Ny)
    
    return image, seeds

def save_sample_to_hdf5_2d(sample_id, tessellation, seeds, filename):
    """
    Save a single 2D Voronoi tessellation sample to an HDF5 file.
    
    Parameters:
    - sample_id: ID of the sample to differentiate between multiple samples.
    - tessellation: The 2D Voronoi tessellation to save.
    - seeds: The seed points corresponding to the tessellation.
    - filename: Name of the HDF5 file to which the data should be saved.
    """
    
    tessellation = np.asarray(tessellation, dtype=np.uint8)
    
    with h5py.File(filename, 'a') as f:
        grp = f.create_group(f'dset_{sample_id}')
        
        compression_opts = 9
        grp.create_dataset('image', data=tessellation, compression="gzip", compression_opts=compression_opts)
        grp.create_dataset('seed_positions', data=seeds, compression="gzip", compression_opts=compression_opts)

def visualize_voronoi_2d(image, seeds=None):
    """
    Visualize a 2D Voronoi tessellation.
    
    Parameters:
    - image: 2D array containing the Voronoi tessellation.
    - seeds: Optional. Original seed points to be plotted over the tessellation.
    """
    plt.imshow(image, cmap='viridis', origin='lower')
    if seeds is not None:
        plt.scatter(seeds[:, 1], seeds[:, 0], c='red', marker='x')
    plt.colorbar()
    plt.title("2D Voronoi Tessellation")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.show()        

from scipy.ndimage import binary_erosion

def periodic_erosion_2d(mask, shrink_factor):
    """Applies erosion on a binary mask in 2D while considering periodic boundaries."""
    selem = np.ones((shrink_factor, shrink_factor))  # 2D structuring element
    padded_mask = np.pad(mask, ((shrink_factor//2, shrink_factor//2), 
                                (shrink_factor//2, shrink_factor//2)), mode='wrap')
    eroded_padded_mask = binary_erosion(padded_mask, structure=selem)
    Nx, Ny = mask.shape
    return eroded_padded_mask[shrink_factor//2:Nx+shrink_factor//2, 
                              shrink_factor//2:Ny+shrink_factor//2]

def shrink_crystals_periodically_2d(image, shrink_factor=5):
    """Erodes (shrinks) each unique crystal in the provided 2D image."""
    unique_crystals = np.unique(image)
    eroded_image = np.full_like(image, 0)  # 0 to indicate eroded space

    for crystal in unique_crystals:
        if crystal == -1:  # Skip the eroded space indicator
            continue
        crystal_mask = (image == crystal)
        eroded_mask = periodic_erosion_2d(crystal_mask, shrink_factor)
        eroded_image[eroded_mask] = 1

    return eroded_image

def save_2d_as_3d_to_hdf5(filename, dsetname, image_2d):
    """
    Save a 2D image as a 3D image in an HDF5 file, by duplicating it along the third dimension.
    
    Parameters:
    - filename: Name of the HDF5 file to which the data should be saved.
    - dsetname: Dataset name under which the image will be saved in the HDF5 file.
    - image_2d: The 2D image to be saved.
    """
    Nx, Ny = image_2d.shape
    # Expanding the 2D image into 3D by duplicating it along the third dimension
    image_3d = np.repeat(image_2d[:, :, np.newaxis], Nx, axis=2)

    with h5py.File(filename, 'a') as f:
        # Check if dataset already exists; if so, replace it
        if dsetname in f:
            del f[dsetname]
        f.create_dataset(dsetname, data=image_3d, dtype=image_2d.dtype, compression='gzip', compression_opts=9)

if __name__ == '__main__':
    # Define parameters for the Voronoi tessellation
    Nx, Ny = 504, 504
    num_crystals_2d = 10
    num_samples = 1
    h5filename_2d = 'voronoi_data_2d.h5'

    # Check if the file already exists and, if so, delete it
    if os.path.exists(h5filename_2d):
        os.remove(h5filename_2d)

    # Sequentially generate, save, and visualize 2D samples
    for i in range(num_samples):
        tessellation, seeds = generate_periodic_voronoi_2d(Nx, Ny, num_crystals_2d)
        # save_sample_to_hdf5_2d(i, tessellation, seeds, h5filename_2d)
        # visualize_voronoi_2d(tessellation, seeds)
        shrink_factor = 25
        eroded_tessellation = shrink_crystals_periodically_2d(tessellation, shrink_factor)
        visualize_voronoi_2d(eroded_tessellation, seeds)
        save_2d_as_3d_to_hdf5("voronoi_eroded2D.h5", "dset_0", eroded_tessellation)
