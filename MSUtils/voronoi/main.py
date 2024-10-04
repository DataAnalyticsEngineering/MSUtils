from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.general.h52xdmf import write_xdmf

from VoronoiGBErosion import *
# from voronoi_helpers import *

if __name__ == "__main__":
    
    num_crystals = 64
    L = [1, 1, 1]

    SeedInfo = VoronoiSeeds(num_crystals, L, "sobol", BitGeneratorSeed=42)
    # SeedInfo.seeds += np.ones_like(SeedInfo.seeds) * 0.25
    # SeedInfo.write("/seeds", "data/voroImg.h5")



    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)
    voroTess.write_to_vtu("data/voroTess.vtu")

    Nx, Ny, Nz = 512, 512, 512    
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, L)
    voroImg.write(h5_filename="data/voroImg.h5", dset_name="/dset_0", order="zyx")
    write_xdmf("data/voroImg.h5", "data/voroImg.xdmf", microstructure_length=[1,1,1])
    
    voroErodedImg = PeriodicVoronoiImageErosion(voroImg, voroTess, shrink_factor=4)
    voroErodedImg.write_to_h5("/dset_0", "data/voroImg_eroded.h5", order="zyx")
    write_xdmf("data/voroImg_eroded.h5", "data/voroImg_eroded.xdmf", microstructure_length=[1,1,1])
    
    
    msimage = MicrostructureImage(image=voroErodedImg.eroded_image, L=L)
    print(msimage.volume_fractions)
