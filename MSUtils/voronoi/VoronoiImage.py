from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree as KDTree

from MSUtils.general.MicrostructureImage import MicrostructureImage


class PeriodicVoronoiImage(MicrostructureImage):
    def __init__(
        self,
        N: npt.ArrayLike = None,
        seeds: int | None = None,
        L: list[float] = None,
        h5_filename: Path = None,
        dset_name: str = None,
    ) -> Self:
        """
        Create a new Periodic Voronoi Image.

        Args:
            N (npt.ArrayLike, optional): _description_. Defaults to None.
            seeds (int|None, optional): _description_. Defaults to None.
            L (list[float], optional): _description_. Defaults to [1, 1, 1].
            h5_filename (Path, optional): _description_. Defaults to None.
            dset_name (str, optional): _description_. Defaults to None.

        Returns:
            Self: The Periodic Voronoi Image
        """
        if L is None:
            L = [1, 1, 1]
        if h5_filename is not None and dset_name is not None:
            super().__init__(h5_filename=h5_filename, dset_name=dset_name)
        else:
            self.seeds = seeds
            self.num_crystals = len(self.seeds)

            image = self._generate_periodic_voronoi(N, L)
            super().__init__(
                h5_filename=h5_filename,
                dset_name=dset_name,
                image=image,
                resolution=N,
                L=L,
            )
            self.compute_volume_fractions()

    def _generate_periodic_voronoi(
        self, N: npt.ArrayLike, L: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        _summary_

        Args:
            N (npt.ArrayLike): _description_
            L (npt.ArrayLike): _description_

        Returns:
            npt.ArrayLike: _description_
        """
        # Adjust for voxel centers and create a mesh grid of points within the main domain
        grid_ranges = [
            np.linspace(0.5 * L[dim] / N[dim], L[dim] - 0.5 * L[dim] / N[dim], N[dim])
            for dim in range(len(L))
        ]
        meshgrid = np.meshgrid(*grid_ranges, indexing="ij")
        points = np.column_stack([axis.ravel() for axis in meshgrid])

        # Use KDTree with periodic boundary conditions to find the nearest seed point for each point in the domain
        tree = KDTree(self.seeds, boxsize=L)
        _, labels = tree.query(points)
        image_shape = tuple(N)
        image = labels.reshape(image_shape)

        return image
