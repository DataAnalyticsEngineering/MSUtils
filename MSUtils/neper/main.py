import subprocess
from pathlib import Path

from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.neper import NeperGBErosion, NeperMicrostructure

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_neper_microstructure(
    h5_filename,
    group_name,
    *,
    neper_executable,
    Nx,
    Ny,
    Nz,
    L,
    num_grains,
    morphology,
    orientation,
    periodicity,
    crystal_symmetry,
    seed,
    interface_thickness,
    tesr_directory="data/neper",
    extra_args=(),
    save_orientations=False,
    save_normals=False,
):
    """Generate a Neper raster tessellation and convert it to HDF5."""
    resolution = (Nx, Ny, Nz)
    group_name = group_name.strip("/")
    if not group_name:
        raise ValueError("group_name cannot be empty.")

    h5_filename = Path(h5_filename)
    h5_filename.parent.mkdir(parents=True, exist_ok=True)
    output_stem = Path(tesr_directory).resolve() / group_name.replace("/", "_")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(neper_executable),
        "-T",
        "-n",
        str(num_grains),
        "-id",
        str(seed),
        "-domain",
        f"cube({','.join(map(str, L))})",
        "-tesrsize",
        ":".join(map(str, resolution)),
        "-morpho",
        morphology,
        "-crysym",
        crystal_symmetry,
        "-ori",
        orientation,
        "-periodicity",
        periodicity,
        "-statface",
        "polys,vernb,vercoos",
        *map(str, extra_args),
        "-format",
        "tess,tesr",
        "-o",
        output_stem.name,
    ]
    subprocess.run(command, check=True, cwd=output_stem.parent)

    microstructure = NeperMicrostructure(output_stem.with_suffix(".tesr"))
    erosion = NeperGBErosion(
        microstructure,
        output_stem.with_suffix(".stface"),
        interface_thickness,
    )
    microstructure.write(h5_filename, f"/{group_name}/microstructure")
    erosion.write_h5(h5_filename, f"/{group_name}", rotation_matrices=microstructure.rotation_matrices, canonical_convention_grains=microstructure.canoncial_convention_grains, save_normals=save_normals, save_orientations=save_orientations)
    return microstructure


def main():
    Nx, Ny, Nz = 256, 256, 256
    L = (1.0, 1.0, 1.0)
    num_grains = 32
    interface_thickness = 6 * L[0] / Nx
    h5_filename = "data/neper_microstructures.h5"
    xdmf_filename = "data/neper_microstructures.xdmf"
    tesr_directory = "data/neper"
    neper_executable = _PROJECT_ROOT / ".pixi/envs/neper/bin/neper"

    examples = (
        {
            # NOTE: no GBs are added for interfaces of self-touching grains!
            "group_name": "diamond",
            "num_grains": "from morpho",
            "morphology": "tocta(1)",
            "periodicity": "all",
            "orientation": "random",
            "crystal_symmetry": "mmm",
        },
        {
            "group_name": "periodic_voronoi",
            "num_grains": num_grains,
            "morphology": "voronoi",
            "periodicity": "all",
            # Use predefined orientations (e.g. same orientation for all grains)
            "orientation": f"file({_PROJECT_ROOT}/data/uniform_orientation.ori,des=rotmat:active)",
            "crystal_symmetry": "mmm",
        },
        {
            "group_name": "centroidal_uniform",
            "num_grains": num_grains,
            "morphology": "centroidal",
            "periodicity": "all",
            "orientation": "random",
            "crystal_symmetry": "cubic",
            "extra_args": ("-morphooptistop", "itermax=100"),
        },
        {
            "group_name": "anisotropic_rolling_texture",
            "num_grains": num_grains,
            "morphology": "gg,aspratio(2,1,0.5)",
            "periodicity": "all",
            "orientation": "Brass1:normal(4)+Copper1:normal(5)+Cube:normal(4)",
            "crystal_symmetry": "cubic",
            "extra_args": ("-morphooptistop", "itermax=100"),
        },
    )

    for seed, example in enumerate(examples, start=1):
        generate_neper_microstructure(
            **example,
            h5_filename=h5_filename,
            neper_executable=neper_executable,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            L=L,
            seed=seed,
            interface_thickness=interface_thickness,
            tesr_directory=tesr_directory,
            save_orientations=True,
            save_normals=True,
        )

    write_xdmf(
        h5_filepath=h5_filename,
        xdmf_filepath=xdmf_filename,
        microstructure_length=L[::-1],
    )


if __name__ == "__main__":
    main()
