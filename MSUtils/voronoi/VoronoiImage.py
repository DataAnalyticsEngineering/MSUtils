import numpy as np
from scipy.spatial import cKDTree as KDTree
import sys, os, numpy as np
# sys.path.append(os.path.join('general/'))
from MSUtils.general.MicrostructureImage import MicrostructureImage

class PeriodicVoronoiImage(MicrostructureImage):
    def __init__(self, N = None, seeds = None, RVE_length=[1, 1, 1], h5_filename=None, dset_name=None):
        if h5_filename is not None and dset_name is not None:
            super().__init__(h5_filename=h5_filename, dset_name=dset_name)
        else:
            self.seeds = seeds
            self.num_crystals = len(self.seeds)

            image = self._generate_periodic_voronoi(N, RVE_length)
            super().__init__(h5_filename=h5_filename, dset_name=dset_name, image=image, resolution=N, L=RVE_length)
            self.compute_volume_fractions()

    def _generate_periodic_voronoi(self, N, L):
        # Adjust for voxel centers and create a mesh grid of points within the main domain
        grid_ranges = [np.linspace(0.5 * L[dim] / N[dim],
                                   L[dim] - 0.5 * L[dim] / N[dim],
                                   N[dim]) for dim in range(len(L))]
        meshgrid = np.meshgrid(*grid_ranges, indexing='ij')
        points = np.column_stack([axis.ravel() for axis in meshgrid])

        # Use KDTree with periodic boundary conditions to find the nearest seed point for each point in the domain
        tree = KDTree(self.seeds, boxsize=L)
        _, labels = tree.query(points)
        image_shape = tuple(N)
        image = labels.reshape(image_shape)

        return image