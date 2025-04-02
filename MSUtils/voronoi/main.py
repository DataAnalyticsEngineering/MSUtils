from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.voronoi.VoronoiGBErosion import PeriodicVoronoiImageErosion
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation


def main():
    num_crystals = 27
    L = [1, 1, 1]
    interface_thickness = 0.02

    SeedInfo = VoronoiSeeds(num_crystals, L, "sobol", BitGeneratorSeed=42)

    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)
    voroTess.write_to_vtu("data/voroTess.vtu")

    Nx, Ny, Nz = 128, 128, 128
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, L)
    voroImg.write(h5_filename="data/voroImg.h5", dset_name="/dset_0", order="zyx")
    write_xdmf("data/voroImg.h5", "data/voroImg.xdmf", microstructure_length=[1, 1, 1])

    voroErodedImg = PeriodicVoronoiImageErosion(voroImg, voroTess, interface_thickness=interface_thickness)
    voroErodedImg.write_h5("data/voroImg_eroded.h5", "/dset_0", order="zyx")
    write_xdmf(
        "data/voroImg_eroded.h5",
        "data/voroImg_eroded.xdmf",
        microstructure_length=[1, 1, 1],
    )

    msimage = MicrostructureImage(image=voroErodedImg.eroded_image, L=L)
    phase_volume_fraction = msimage.volume_fractions.get(-1, 0) * 100
    print(f"Volume fraction of grain boundary (phase: -1): {phase_volume_fraction:.4f}%")


if __name__ == "__main__":
    main()
