from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.voronoi.VoronoiGBErosion import PeriodicVoronoiImageErosion
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation

def main():
    num_crystals = 27
    L = [1, 1, 1]
    Nx, Ny, Nz = 256, 256, 256
    permute_order = "zyx"

    # Generate Voronoi seeds and tessellation
    SeedInfo = VoronoiSeeds(num_crystals, L, "sobol", BitGeneratorSeed=42)
    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)
    voroTess.write_to_vtu("data/voroTess.vtu")

    # Generate Voronoi image
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, L)
    voroImg.write(
        h5_filename="data/voroImg.h5", dset_name="/dset_0", order=permute_order
    )
    write_xdmf("data/voroImg.h5", "data/voroImg.xdmf", microstructure_length=[1, 1, 1])

    # Generate Voronoi image with grain boundaries of a specific thickness
    interface_thickness = (1.0 / 256) * 6
    voroErodedImg = PeriodicVoronoiImageErosion(
        voroImg, voroTess, interface_thickness=interface_thickness
    )
    voroErodedImg.write_h5("data/voroImg_eroded.h5", "/dset_0", order=permute_order)
    write_xdmf(
        "data/voroImg_eroded.h5",
        "data/voroImg_eroded.xdmf",
        microstructure_length=[1, 1, 1],
    )
    
    # Calculate and print volume fraction of all grain boundary (all tags >= num_crystals)
    msimage = MicrostructureImage(image=voroErodedImg.eroded_image, L=L)
    gb_volume_fraction = 0
    for phase, fraction in msimage.volume_fractions.items():
        if phase >= num_crystals:
            gb_volume_fraction += fraction
    
    gb_volume_fraction_percent = gb_volume_fraction * 100
    print(f"Volume fraction of all grain boundaries: {gb_volume_fraction_percent:.8f}%")
    print(f"Interface thickness: {interface_thickness:.10f}")

if __name__ == "__main__":
    main()
