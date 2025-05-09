from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.voronoi.VoronoiGBErosion import PeriodicVoronoiImageErosion
from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation

import json
def main():
    num_crystals = 8
    L = [1, 1, 1]
    Nx, Ny, Nz = 128, 128, 128
    permute_order = "zyx"

    # Generate Voronoi seeds and tessellation
    SeedInfo = VoronoiSeeds(num_crystals, L, "diamond", BitGeneratorSeed=42)
    voroTess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)
    # voroTess.write_to_vtu("data/voroTess.vtu")

    # Generate Voronoi image
    voroImg = PeriodicVoronoiImage([Nx, Ny, Nz], SeedInfo.seeds, L)
    # voroImg.write(
    #     h5_filename="data/voroImg.h5", dset_name="/dset_0", order=permute_order
    # )
    # write_xdmf("data/voroImg.h5", "data/voroImg.xdmf", microstructure_length=[1, 1, 1])

    # Generate Voronoi image with grain boundaries of a specific thickness
    interface_thickness = (1.0 / 128) * 6
    voroErodedImg = PeriodicVoronoiImageErosion(
        voroImg, voroTess, interface_thickness=interface_thickness
    )
    voroErodedImg.write_h5("data/voroImg_eroded.h5", "/dset_0", order=permute_order)
    write_xdmf(
        "data/voroImg_eroded.h5",
        "data/voroImg_eroded.xdmf",
        microstructure_length=[1, 1, 1],
    )
    
    # # Print human-readable format
    # for ridge_tag in sorted(voroErodedImg.ridge_metadata.keys()):
    #     normal, crystalA, crystalB = voroErodedImg.ridge_metadata[ridge_tag]
    #     print(f"Ridge {ridge_tag} is between crystal {crystalA}, crystalB {crystalB} with normal={normal}")
    
    # # Create JSON structure
    # gb_info = {}
    # for ridge_tag, (normal, _, _) in voroErodedImg.ridge_metadata.items():
    #     gb_info[str(ridge_tag)] = {
    #         "normal": normal.tolist()
    #     }
    
    # # dump the JSON structure to a file
    # with open("data/ridge_metadata_dump.json", "w") as f:
    #     json.dump(gb_info, f, indent=4)
    
    # # # Create the outer structure with "GBInfo" key
    # # output_json = {"GBVoxelInfo": gb_info}
    
    # # Write to JSON file with custom formatting
    # with open("data/ridge_metadata.json", "w") as f:
    #     # Write the opening brace and GBInfo key
    #     f.write("{\n  \"GBVoxelInfo\": {\n")
        
    #     # Sort keys to ensure consistent ordering
    #     sorted_keys = sorted(gb_info.keys(), key=lambda x: int(x))
        
    #     # Write each GB entry on its own line
    #     for i, key in enumerate(sorted_keys):
    #         entry = gb_info[key]
    #         comma = "," if i < len(sorted_keys) - 1 else ""
    #         f.write(f"    \"{key}\": {{ \"normal\": {entry['normal']}}}{comma}\n")
        
    #     # Write closing braces
    #     f.write("  }\n}")
    
    # print(f"Ridge metadata written to data/ridge_metadata.json")

if __name__ == "__main__":
    main()
