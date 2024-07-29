import numpy as np
import h5py
from scipy.stats import qmc 

class VoronoiSeeds:
    def __init__(self, num_crystals=None, RVE_length=[1, 1, 1], method="sobol", BitGeneratorSeed=None, filename=None, grp_name=None):
        """
        Generate seed points (crystals) for 3D Voronoi tessellations and their associated orthonormal lattice vectors.

        Parameters:
        - num_crystals: Number of seed points (or crystals) to generate.
        - RVE_length: Dimensions of the representative volume element (RVE).
        - method: Method to generate seed points. Can be one of the following: "random", "lhs-llyod", "halton", "sobol".
        - BitGeneratorSeed: Optional. Seed for the random number generator.
        - filename: Optional. HDF5 file to read data from.
        - grp_name: Optional. Group name in the HDF5 file to read data from.
       
        """

        if filename and grp_name:
            self.read(filename, grp_name)
        else:
            if num_crystals is None:
                raise ValueError("num_crystals must be specified if not reading from file.")
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

    def _generate_seeds(self):
        """
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
        dim = len(self.RVE_length)  # Determine the dimension (2D or 3D)
        rng = np.random.default_rng(self.BitGeneratorSeed)  # Create a random number generator with the specified seed

        # Generate seeds based on method
        if self.method == "random":
            self.seeds = rng.random((self.num_crystals, dim)) * np.array(self.RVE_length)
        elif self.method == "lhs-lloyd":
            sampler = qmc.LatinHypercube(d=dim, seed=self.BitGeneratorSeed, optimization='lloyd')
            self.seeds = sampler.random(n=self.num_crystals) * np.array(self.RVE_length)
        elif self.method == "halton":
            sampler = qmc.Halton(d=dim, seed=self.BitGeneratorSeed)
            self.seeds = sampler.random(n=self.num_crystals) * np.array(self.RVE_length)
        elif self.method == "sobol":
            sampler = qmc.Sobol(d=dim, seed=self.BitGeneratorSeed)
            self.seeds = sampler.random_base2(m=int(np.ceil(np.log2(self.num_crystals))))[:self.num_crystals] * np.array(self.RVE_length)
        else:
            raise ValueError("Unknown sampling method!") 
            # TODO: Add more sampling methods here using PeriodicVoronoiTessellation class 


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

        rot_matrices = np.stack((R11, R12, R13, R21, R22, R23, R31, R32, R33), axis=-1).reshape(self.num_crystals, 3, 3)

        # The initial orientation is simply an identity matrix, representing [1, 0, 0], [0, 1, 0], and [0, 0, 1]
        # Using the rotation matrices, we compute the lattice vectors for each seed
        initial_orientation = np.eye(3)
        self.lattice_vectors = np.einsum('nij,jk->nik', rot_matrices, initial_orientation)

    def write(self, grp_name, filename):
        with h5py.File(filename, 'a') as h5_file:
            # Check if the group already exists and delete it
            if grp_name in h5_file:
                del h5_file[grp_name]

            # Create a new group
            grp = h5_file.create_group(grp_name)
            compression_opts = 9

            # Create the datasets and write the data
            grp.create_dataset('seed_positions', data=self.seeds, dtype=np.float64, compression="gzip", compression_opts=compression_opts)
            grp.create_dataset('lattice_vectors', data=self.lattice_vectors, dtype=np.float64, compression="gzip", compression_opts=compression_opts)
            grp.create_dataset('Microstructure_length', data=self.RVE_length, dtype=np.float64, compression="gzip", compression_opts=compression_opts)

    def read(self, filename, grp_name):
        with h5py.File(filename, 'r') as h5_file:
            if grp_name not in h5_file:
                raise ValueError(f"Group {grp_name} not found in file {filename}.")
            grp = h5_file[grp_name]

            self.seeds = grp['seed_positions'][:]
            self.lattice_vectors = grp['lattice_vectors'][:]
            self.RVE_length = grp['Microstructure_length'][:]
            self.num_crystals = self.seeds.shape[0]

def test_sampling_methods(methods):
    num_crystals = 1024
    RVE_length = [1.0, 1.0, 1.0]
    BitGeneratorSeed = 42
    nbins = 128

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from VoronoiTessellation import PeriodicVoronoiTessellation

    fig = make_subplots(rows=1, cols=1)
    for method in methods:
        seeds = VoronoiSeeds(num_crystals=num_crystals, RVE_length=RVE_length, method=method, BitGeneratorSeed=BitGeneratorSeed)
        voroTess = PeriodicVoronoiTessellation(RVE_length, seeds.seeds)
        
        volumes = voroTess.crystal_volumes
        hist = go.Histogram(
            x=volumes,
            nbinsx=nbins,
            name=method,
            opacity=1.0
        )
        fig.add_trace(hist)

    fig.update_layout(
        title="Grain volume fraction histogram",
        xaxis_title="Grain volume fraction",
        yaxis_title="Frequency",
        legend=dict(x=1.05, y=1)
    )
    fig.show()

if __name__ == "__main__":
    # Example parameters
    num_crystals = 32
    RVE_length = [1.0, 1.0, 1.0]
    method = "random"
    BitGeneratorSeed = 42
    grp_name = "voronoi_seeds_group"
    filename = "data/voronoi_seeds.h5"

    # Create an instance of VoronoiSeeds
    seeds = VoronoiSeeds(num_crystals=num_crystals, RVE_length=RVE_length, method=method, BitGeneratorSeed=BitGeneratorSeed)

    # Write the seeds to an HDF5 file
    seeds.write(grp_name=grp_name, filename=filename)

    # Test sampling methods
    methods = ["random", "sobol"]
    test_sampling_methods(methods)
