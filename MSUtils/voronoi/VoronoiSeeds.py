from pathlib import Path
from typing import Self

import h5py
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

from MSUtils.voronoi.voronoi_helpers import factorize


class VoronoiSeeds:
    def __init__(
        self,
        num_crystals: int = None,
        RVE_length: list[float] = None,
        method: str = "sobol",
        BitGeneratorSeed: int = None,
        filename: str = None,
        grp_name: str = None,
    ) -> Self:
        """
        Class to handle Voronoi seed points.

        Generate seed points (crystals) for 3D Voronoi tessellations and their associated orthonormal lattice vectors.

        Args:
            num_crystals (int, optional):  Number of seed points (or crystals) to generate.. Defaults to None.
            RVE_length (list[float], optional): Dimensions of the representative volume element (RVE). Defaults to [1, 1, 1].
            method (str, optional): Method to generate seed points. Defaults to "sobol".
            BitGeneratorSeed (int, optional): Seed for the random number generator.. Defaults to None.
            filename (str, optional): HDF5 file to read data from. Defaults to None.
            grp_name (str, optional): Group name in the HDF5 file to read data from. Defaults to None.

        Raises:
            ValueError: num_crystals must be specified if not reading from file.
            ValueError: Number of crystals must be positive.
            ValueError: RVE_length must have 2 or 3 dimensions.

        Returns:
            Self: Voronoi Seeds Object

        Notes:

            - method can be one of the following: "random", "lhs-llyod", "halton", "sobol", "honeycomb", "rubiks-cube", "diamond".
        """
        if RVE_length is None:
            RVE_length = [1, 1, 1]
        if filename and grp_name:
            self.read_h5(filename, grp_name)
        else:
            if num_crystals is None:
                raise ValueError(
                    "num_crystals must be specified if not reading from file."
                )
            if num_crystals <= 0:
                raise ValueError("Number of crystals must be positive.")
            if len(RVE_length) not in [2, 3]:
                raise ValueError("RVE_length must have 2 or 3 dimensions.")

            self.num_crystals = num_crystals
            self.RVE_length = RVE_length
            self.method = method
            self.BitGeneratorSeed = BitGeneratorSeed
            self.seeds = None
            self.lattice_vectors = None
            self._generate_seeds()

    def _generate_seeds(self) -> None:
        """
        _summary_

        Raises:
            ValueError: _description_

        Notes:
            A uniform distribution on the group of rotations SO(3) is used to assign random
            orientations to the grains of the aggregate. This is based on random variables X, Y, Z
            that are uniformly distributed on the interval [0, 1) to determine three Euler angles
            (z-x-z convention) via ϕ1 = 2πX, Φ = acos(2Y - 1), ϕ2 = 2πZ.
        """
        # This is old, no idea what it means...
        # Returns:
        # - seeds: Generated seed points of shape (num_crystals, 3).
        # - lattice_vectors: Orthonormal lattice vectors for each seed. The array has a shape
        # of (num_crystals, 3, 3), where:
        #     * The first dimension corresponds to each crystal.
        #     * The second dimension enumerates the three orthonormal vectors for each crystal.
        #     * The third dimension represents the three components of a vector (x, y, z).

        dim = len(self.RVE_length)  # Determine the dimension (2D or 3D)
        rng = np.random.default_rng(
            self.BitGeneratorSeed
        )  # Create a random number generator with the specified seed

        # Generate seeds based on method
        match self.method:
            case "random":
                self.seeds = rng.random((self.num_crystals, dim)) * np.array(
                    self.RVE_length
                )
            case "lhs-lloyd":
                sampler = qmc.LatinHypercube(
                    d=dim, seed=self.BitGeneratorSeed, optimization="lloyd"
                )
                self.seeds = sampler.random(n=self.num_crystals) * np.array(
                    self.RVE_length
                )
            case "halton":
                sampler = qmc.Halton(d=dim, seed=self.BitGeneratorSeed)
                self.seeds = sampler.random(n=self.num_crystals) * np.array(
                    self.RVE_length
                )
            case "sobol":
                sampler = qmc.Sobol(d=dim, seed=self.BitGeneratorSeed)
                self.seeds = sampler.random_base2(
                    m=int(np.ceil(np.log2(self.num_crystals)))
                )[: self.num_crystals] * np.array(self.RVE_length)
            case "honeycomb":
                N = factorize(self.num_crystals, dim)
                self.seeds = self._generate_lattice(
                    N[0], N[1], N[2], self.RVE_length, stagger=True
                )
            case "rubiks-cube":
                N = factorize(self.num_crystals, dim)
                self.seeds = self._generate_lattice(
                    N[0], N[1], N[2], self.RVE_length, stagger=False
                )
            case "diamond":  # Diamond-like seed points
                self.num_crystals = 2
                self.seeds = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
            case _:
                raise ValueError("Unknown sampling method! : " + self.method)
                # Add more sampling methods here if needed

        # Generate random orientations using the Euler angles (z-x-z convention)
        X, Y, Z = np.random.rand(3, self.num_crystals)
        phi1 = 2 * np.pi * X
        Phi = np.arccos(2 * Y - 1)
        phi2 = 2 * np.pi * Z

        # Convert Euler angles to rotation matrices
        c1, s1 = np.cos(phi1), np.sin(phi1)
        cP, sP = np.cos(Phi), np.sin(Phi)
        c2, s2 = np.cos(phi2), np.sin(phi2)

        R11 = c1 * c2 - cP * s1 * s2
        R12 = -c1 * s2 - cP * c2 * s1
        R13 = s1 * sP
        R21 = c2 * s1 + c1 * cP * s2
        R22 = c1 * cP * c2 - s1 * s2
        R23 = -c1 * sP
        R31 = sP * s2
        R32 = c2 * sP
        R33 = cP

        rot_matrices = np.stack(
            (R11, R12, R13, R21, R22, R23, R31, R32, R33), axis=-1
        ).reshape(self.num_crystals, 3, 3)

        # Using the rotation matrices, we compute the lattice vectors for each seed
        initial_orientation = np.eye(3)
        self.lattice_vectors = np.einsum(
            "nij,jk->nik", rot_matrices, initial_orientation
        )

    def _generate_lattice(
        self, Nx: int, Ny: int, Nz: int, RVE_length: list[float], stagger: bool = True
    ) -> npt.ArrayLike:
        """
        Return points in a 3D lattice with optional staggering, ensuring symmetry across the xy, xz, and yz planes.

        Args:
            Nx (int): Number of points along the x-axis.
            Ny (int): Number of points along the y-axis.
            Nz (int): Number of points along the z-axis.
            RVE_length (list[float]): Lengths of the box in each dimension (x, y, z).
            stagger (bool, optional): Boolean indicating whether to apply staggering. Defaults to True.

        Returns:
            npt.ArrayLike: Seed points in lattice arrangement.
        """
        points = []
        a = RVE_length[0] / Nx
        b = RVE_length[1] / Ny
        c = RVE_length[2] / Nz

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # Generate base lattice points
                    x = i * a
                    y = j * b
                    z = k * c
                    if stagger:
                        # Apply staggering in x, y, and z directions
                        if j % 2 == 1:
                            x += a / 2  # Stagger every other row in x direction
                        if k % 2 == 1:
                            y += b / 2  # Stagger every other layer in y direction
                            x += a / 2  # Further stagger in x direction
                    # Add the point, ensuring it's within the RVE box
                    points.append(
                        [x % RVE_length[0], y % RVE_length[1], z % RVE_length[2]]
                    )
        return np.array(points)

    def write_h5(self, filepath: Path, grp_name: str) -> None:
        """
        Write Seeds into h5py file.

        Args:
            filepath (Path): Filepath
            grp_name (str): Name of h5py group

        Notes:
            The tree looks like this:
            grp_name
             |- seed_positions
             |- lattice_vectors
             |- Microstructure_length
        """
        with h5py.File(filepath, "a") as h5_file:
            # Check if the group already exists and delete it
            if grp_name in h5_file:
                del h5_file[grp_name]

            # Create a new group
            grp = h5_file.create_group(grp_name)
            compression_opts = 9

            # Create the datasets and write the data
            grp.create_dataset(
                "seed_positions",
                data=self.seeds,
                dtype=np.float64,
                compression="gzip",
                compression_opts=compression_opts,
            )
            grp.create_dataset(
                "lattice_vectors",
                data=self.lattice_vectors,
                dtype=np.float64,
                compression="gzip",
                compression_opts=compression_opts,
            )
            grp.create_dataset(
                "Microstructure_length",
                data=self.RVE_length,
                dtype=np.float64,
                compression="gzip",
                compression_opts=compression_opts,
            )

    def read_h5(self, filepath: Path, grp_name: str) -> None:
        """
        Read in Voronoi-Seeds from a h5 file.

        Args:
            filepath (Path): Filepath
            grp_name (str): Name of h5py group

        Raises:
            ValueError: Group not found in file.
        """
        with h5py.File(filepath, "r") as h5_file:
            if grp_name not in h5_file:
                raise ValueError(f"Group {grp_name} not found in file {filepath}.")
            grp = h5_file[grp_name]

            self.seeds = grp["seed_positions"][:]
            self.lattice_vectors = grp["lattice_vectors"][:]
            self.RVE_length = grp["Microstructure_length"][:]
            self.num_crystals = self.seeds.shape[0]


def compare_sampling_methods(methods):
    num_crystals = 1000
    RVE_length = [1.0, 1.0, 1.0]
    BitGeneratorSeed = 42
    nbins = 128

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation

    fig = make_subplots(rows=1, cols=1)
    for method in methods:
        seeds = VoronoiSeeds(
            num_crystals=num_crystals,
            RVE_length=RVE_length,
            method=method,
            BitGeneratorSeed=BitGeneratorSeed,
        )
        voroTess = PeriodicVoronoiTessellation(RVE_length, seeds.seeds)

        volumes = voroTess.crystal_volumes
        hist = go.Histogram(x=volumes, nbinsx=nbins, name=method, opacity=1.0)
        fig.add_trace(hist)

    fig.update_layout(
        title="Grain volume fraction histogram",
        xaxis_title="Grain volume fraction",
        yaxis_title="Frequency",
        legend=dict(x=1.05, y=1),
        autosize=False,
        width=1400,
        height=900,
        font=dict(size=20),
        template="simple_white",
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",  # Set the grid color to make it visible
        layer="above traces",  # Bring grid lines to the front
        mirror=True,
        ticks="inside",
        tickwidth=2,
        ticklen=6,
        title_font=dict(size=20),
        tickfont=dict(size=20),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",  # Set the grid color to make it visible
        layer="above traces",  # Bring grid lines to the front
        mirror=True,
        ticks="inside",
        tickwidth=2,
        ticklen=6,
        title_font=dict(size=20),
        tickfont=dict(size=20),
        automargin=True,
    )
    fig.show()


def main():
    # Example parameters
    num_crystals = 1024
    RVE_length = [1.0, 1.0, 1.0]
    method = "sobol"
    BitGeneratorSeed = 42
    grp_name = "voronoi_seeds_group"
    filename = "data/voronoi_seeds.h5"

    # Create an instance of VoronoiSeeds
    seeds = VoronoiSeeds(
        num_crystals=num_crystals,
        RVE_length=RVE_length,
        method=method,
        BitGeneratorSeed=BitGeneratorSeed,
    )

    # Write the seeds to an HDF5 file
    seeds.write_h5(grp_name=grp_name, filepath=filename)

    # Test sampling methods
    methods = ["random", "lhs-lloyd", "halton", "sobol", "honeycomb", "rubiks-cube"]
    compare_sampling_methods(methods)


if __name__ == "__main__":
    main()
