import subprocess
from pathlib import Path

import h5py
import numpy as np

from MSUtils.general.grid import GridSpec
from MSUtils.general.MicrostructureImage import MicrostructureImage


def _line(file) -> str:
    return file.readline().decode().strip()


def _expect(file, expected: str) -> None:
    if _line(file) != expected:
        raise ValueError(f"Expected {expected!r} in {file.name}.")


def _read_faces(filename: str | Path):
    with Path(filename).open(encoding="utf-8") as file:
        for line in file:
            values = line.split()
            poly_a, poly_b, vertex_count = map(int, values[:3])
            vertices = np.asarray(values[3:], dtype=float).reshape(vertex_count, 3)
            yield poly_a, poly_b, vertices


class NeperMicrostructure(MicrostructureImage):
    """Generate and read a Neper 5 raster tessellation."""

    def __init__(
        self,
        output_stem,
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
        extra_args=(),
    ):
        output_stem = Path(output_stem).resolve()
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
            f"{Nx}:{Ny}:{Nz}",
            "-tesrformat",
            "binary16",
            "-morpho",
            morphology,
            "-crysym",
            crystal_symmetry,
            "-ori",
            orientation,
            "-oridescriptor",
            "rotmat:active",
            "-periodicity",
            periodicity,
            "-statface",
            "polys,vernb,vercoos",
            "-statcell",
            "coo,vol,area,sphericity,facenb",
            *map(str, extra_args),
            "-format",
            "tesr,obj",
            "-o",
            output_stem.name,
        ]
        subprocess.run(command, check=True, cwd=output_stem.parent)

        self.tesr_filename = output_stem.with_suffix(".tesr")
        with self.tesr_filename.open("rb") as file:
            for marker in ("***tesr", "**format", "2.2", "**general", "3"):
                _expect(file, marker)

            self.resolution = tuple(map(int, _line(file).split()))
            self.voxel_size = tuple(map(float, _line(file).split()))
            self.origin = (0.0, 0.0, 0.0)

            _expect(file, "**cell")
            self.grain_count = int(_line(file))
            _expect(file, "*id")
            grain_ids = []
            while len(grain_ids) < self.grain_count:
                grain_ids.extend(map(int, _line(file).split()))
            self.grain_ids = np.asarray(grain_ids)

            _expect(file, "*ori")
            _expect(file, "rotmat:active")
            self.rotation_matrices = np.array(
                [_line(file).split() for _ in range(self.grain_count)], dtype=float
            ).reshape(self.grain_count, 3, 3)

            _expect(file, "*crysym")
            self.crystal_symmetry = _line(file)
            _expect(file, "**data")
            _expect(file, "binary16")

            voxel_count = int(np.prod(self.resolution))
            self.image = np.fromfile(file, dtype="<u2", count=voxel_count)
            self.image = np.ascontiguousarray(
                self.image.reshape(self.resolution, order="F"), dtype=np.int32
            )

        self.void_present = bool(np.any(self.image == 0))
        self.image += int(self.void_present) - 1
        if self.void_present:
            self.rotation_matrices = np.concatenate(
                (np.eye(3)[None], self.rotation_matrices)
            )

        lengths = tuple(
            size * spacing for size, spacing in zip(self.resolution, self.voxel_size)
        )
        super().__init__(
            image=self.image,
            grid=GridSpec(shape=self.resolution, lengths=lengths),
        )
        self._characterize(output_stem)

    def _characterize(self, output_stem: Path) -> None:
        stats = np.loadtxt(output_stem.with_suffix(".stcell"), ndmin=2)
        self.crystal_centroids = stats[:, :3]  # Cell centroids.
        self.crystal_volumes = stats[:, 3]  # Exact cell volumes.
        self.crystal_surface_areas = stats[:, 4]  # Total cell surface areas.
        self.crystal_sphericities = stats[:, 5]  # Equal-volume sphere area / area.
        self.crystal_face_counts = stats[:, 6].astype(int)  # Faces per cell.

        self.interface_area = 0.0
        self.Ltensor = np.zeros((3, 3))  # Area-weighted second normal moment.
        for _, _, vertices in _read_faces(output_stem.with_suffix(".stface")):
            area_vector = 0.5 * np.sum(
                np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0
            )
            area = np.linalg.norm(area_vector)
            normal = area_vector / area
            self.interface_area += area
            self.Ltensor += area * np.einsum("i,j->ij", normal, normal)

    def write_h5(
        self,
        h5_filename: str | Path,
        grp_name: str,
        order: str = "zyx",
        compression_level: int = 6,
    ) -> None:
        """Write the microstructure and its rotation matrices."""
        group_name = grp_name.strip("/")
        super().write(
            h5_filename, f"{group_name}/microstructure", order, compression_level
        )

        with h5py.File(self.h5_filename, "a") as h5_file:
            group = h5_file[group_name or "/"]
            if "rotation_matrices" in group:
                del group["rotation_matrices"]
            group.create_dataset("rotation_matrices", data=self.rotation_matrices)
