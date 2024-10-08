# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.voronoi.VoronoiGBErosion import PeriodicVoronoiImageErosion
import numpy as np

# %%
if __name__ == "__main__":
    
    num_crystals = 8
    L = [1, 1, 1]

    SeedInfo = VoronoiSeeds(num_crystals, L, "rubiks-cube", BitGeneratorSeed=42)
    SeedInfo.write("/seeds", "data/voroImg.h5")

    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)
    voroTess.write_to_vtu("data/voroTess.vtu")

    Nx, Ny, Nz = 64, 64, 64    
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, L)
    voroImg.write(h5_filename="data/voroImg.h5", dset_name="/dset_0", order="zyx")
    write_xdmf("data/voroImg.h5", "data/voroImg.xdmf", microstructure_length=[1,1,1])
    
    voroErodedImg = PeriodicVoronoiImageErosion(voroImg, voroTess, shrink_factor=2)
    voroErodedImg.write("/dset_0", "data/voroImg_eroded.h5", order="zyx")
    write_xdmf("data/voroImg_eroded.h5", "data/voroImg_eroded.xdmf", microstructure_length=[1,1,1])
    
    
    msimage = MicrostructureImage(image=voroErodedImg.eroded_image, L=L)
    print(msimage.volume_fractions)

# %%
