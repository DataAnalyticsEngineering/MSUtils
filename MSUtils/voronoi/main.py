from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds

# from GBErosion import *
# from voronoi_helpers import *

if __name__ == "__main__":
    
    num_crystals = 1024
    RVE_length = [1, 1, 1]

    SeedInfo = VoronoiSeeds(num_crystals, RVE_length, "sobol", BitGeneratorSeed=42)
    SeedInfo.write("/seeds", "data/voroImg.h5")

    # SeedInfo.seeds = np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75]]) #+ np.random.normal(loc=0.0, scale=0.05, size=(8, 3))
    # SeedInfo.seeds = np.array([[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]])
    # SeedInfo.seeds = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    # SeedInfo.seeds = np.array([[0.5, 0.5, 0.5]] )


    voroTess = PeriodicVoronoiTessellation(RVE_length, SeedInfo.seeds)
    voroTess.write_to_vtu("data/voroTess.vtu")

    Nx, Ny, Nz = 256, 256, 256    
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, RVE_length)
    voroImg.write(h5_filename="data/voroImg.h5", dset_name="/dset_0", order="zyx")





    # voroErodedImg = PeriodicVoronoiImageErosion(voroImg, voroTess, shrink_factor=2)
    # voroErodedImg.write_to_h5("/dset_0", h5filename, order="zyx")
    




    
    # save_sample_to_hdf5("/dset_0", tessellation=voroImg, filename=h5filename, seeds=seeds, neighbors=neighbors)


    # crystal_index = 231
    # mask = np.isin(voroImg, [crystal_index] + neighbors[crystal_index])
    # tessellation = np.where(mask, voroImg, 0)
    # save_sample_to_hdf5("/dset_1", tessellation, h5filename, seeds, neighbors=neighbors)