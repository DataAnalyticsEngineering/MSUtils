from pathlib import Path

from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.neper import NeperGBErosion, NeperMicrostructure

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    Nx, Ny, Nz = 256, 256, 256
    L = (1.0, 1.0, 1.0)
    num_grains = 32
    interface_thickness = 6 * L[0] / Nx
    h5_filename = Path("data/neper_microstructures.h5")
    xdmf_filename = "data/neper_microstructures.xdmf"
    tesr_directory = Path("data/neper")
    neper_executable = _PROJECT_ROOT / ".pixi/envs/neper/bin/neper"

    examples = {
        "diamond": {
            "num_grains": "from morpho",
            "morphology": "tocta(1)",
            "periodicity": "all",
            "orientation": "random",
            "crystal_symmetry": "mmm",
        },
        "periodic_voronoi": {
            "num_grains": num_grains,
            "morphology": "voronoi",
            "periodicity": "all",
            "orientation": "random",
            "crystal_symmetry": "cubic",
        },
        "centroidal_uniform": {
            "num_grains": num_grains,
            "morphology": "centroidal",
            "periodicity": "all",
            "orientation": "random",
            "crystal_symmetry": "cubic",
            "extra_args": ("-morphooptistop", "itermax=100"),
        },
        "anisotropic_rolling_texture": {
            "num_grains": num_grains,
            "morphology": "gg,aspratio(2,1,0.5)",
            "periodicity": "all",
            "orientation": "Brass1:normal(4)+Copper1:normal(5)+Cube:normal(4)",
            "crystal_symmetry": "cubic",
            "extra_args": ("-morphooptistop", "itermax=100"),
        },
    }

    for seed, (group_name, parameters) in enumerate(examples.items(), start=1):
        microstructure = NeperMicrostructure(
            tesr_directory / group_name,
            **parameters,
            neper_executable=neper_executable,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            L=L,
            seed=seed,
        )
        microstructure.write_h5(h5_filename, group_name)

        erosion = NeperGBErosion(microstructure, interface_thickness)
        erosion.write_h5(
            h5_filename,
            group_name,
            save_normals=True,
            save_orientations=False,
        )

    write_xdmf(
        h5_filepath=h5_filename,
        xdmf_filepath=xdmf_filename,
        microstructure_length=L[::-1],
    )


if __name__ == "__main__":
    main()
