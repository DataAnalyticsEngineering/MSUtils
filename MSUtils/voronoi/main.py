from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.general.h52xdmf import write_xdmf

from VoronoiGBErosion import *
# from voronoi_helpers import *

if __name__ == "__main__":
    
    num_crystals = 8
    L = [1, 1, 1]

    SeedInfo = VoronoiSeeds(num_crystals, L, "rubiks-cube", BitGeneratorSeed=42)
    SeedInfo.seeds += np.ones_like(SeedInfo.seeds) * 0.25
    # SeedInfo.write("/seeds", "data/voroImg.h5")



    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)
    voroTess.write_to_vtu("data/voroTess.vtu")

    Nx, Ny, Nz = 64, 64, 64    
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, L)
    voroImg.write(h5_filename="data/voroImg.h5", dset_name="/dset_0", order="zyx")
    write_xdmf("data/voroImg.h5", "data/voroImg.xdmf", microstructure_length=[1,1,1])





    voroErodedImg = PeriodicVoronoiImageErosion(voroImg, voroTess, shrink_factor=2)
    voroErodedImg.write_to_h5("/dset_0", "data/voroImg_eroded.h5", order="zyx")
    write_xdmf("data/voroImg_eroded.h5", "data/voroImg_eroded.xdmf", microstructure_length=[1,1,1])
    






















    # SeedInfo.seeds = np.array([
    #     [0.25, 0.25, 0.25], 
    #     [0.25, 0.25, 0.75], 
    #     [0.25, 0.75, 0.25], 
    #     [0.25, 0.75, 0.75], 
    #     [0.75, 0.25, 0.25], 
    #     [0.75, 0.25, 0.75], 
    #     [0.75, 0.75, 0.25], 
    #     [0.75, 0.75, 0.75]
    # ])  # + np.random.normal(loc=0.0, scale=0.01, size=(8, 3))
    
    # SeedInfo.seeds = np.array([[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]])
    # SeedInfo.seeds = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    # SeedInfo.seeds = np.array([[0.5, 0.5, 0.5]] )






    
    # save_sample_to_hdf5("/dset_0", tessellation=voroImg, filename=h5filename, seeds=seeds, neighbors=neighbors)


    # crystal_index = 231
    # mask = np.isin(voroImg, [crystal_index] + neighbors[crystal_index])
    # tessellation = np.where(mask, voroImg, 0)
    # save_sample_to_hdf5("/dset_1", tessellation, h5filename, seeds, neighbors=neighbors)