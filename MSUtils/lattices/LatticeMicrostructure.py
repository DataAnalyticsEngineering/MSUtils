from pathlib import Path

import numpy as np

from MSUtils.general.draw_strut import draw_strut
from MSUtils.general.grid import GridSpec
from MSUtils.general.MicrostructureImage import MicrostructureImage


class LatticeMicrostructure(MicrostructureImage):
    """Rasterize a periodic strut lattice on a cell-centered grid."""

    def __init__(self, *, Nx, Ny, Nz, L, radius, unit_cell):
        grid = GridSpec(shape=(Nx, Ny, Nz), lengths=L)
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive and finite.")

        vertices, edges = unit_cell()
        vertices = np.asarray(vertices, dtype=float)
        edges = np.asarray(edges)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError("Unit-cell vertices must have shape (n, 3) and be finite.")
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=int)
        elif (
            edges.ndim != 2
            or edges.shape[1] != 2
            or not np.issubdtype(edges.dtype, np.integer)
        ):
            raise ValueError("Unit-cell edges must contain pairs of vertex indices.")
        if np.any(edges < 0) or np.any(edges >= len(vertices)):
            raise ValueError("Unit-cell edge index is out of bounds.")
        if np.any(np.all(vertices[edges[:, 0]] == vertices[edges[:, 1]], axis=1)):
            raise ValueError("Unit-cell edges must have nonzero length.")

        image = np.zeros(grid.shape, dtype=np.uint8)
        vertices = vertices * np.asarray(grid.lengths)
        for start, end in vertices[edges]:
            draw_strut(image, start, end, radius, grid, periodic=True)

        self.radius = radius
        super().__init__(
            image=image,
            grid=grid,
            metadata={
                "lattice_type": unit_cell.__name__,
                "strut_radius": radius,
            },
        )

    def write_h5(
        self,
        h5_filename: str | Path,
        grp_name: str,
        order: str = "zyx",
        compression_level: int = 6,
    ) -> None:
        super().write(
            h5_filename,
            f"{grp_name.strip('/')}/microstructure",
            order,
            compression_level,
        )


def main():
    from MSUtils.general.h52xdmf import write_xdmf
    from MSUtils.lattices.lattice_definitions import (
        BCC_lattice,
        BCCz_lattice,
        FBCC_lattice,
        FCC_lattice,
        auxetic_lattice,
        cubic_lattice,
        isotruss_lattice,
        octet_truss_lattice,
    )

    Nx, Ny, Nz = 400, 400, 400
    L = (1.0, 1.0, 1.0)
    radius = 0.05
    h5_filename = Path("data/lattice_microstructures.h5")
    unit_cells = {
        "BCC": BCC_lattice,
        "BCCz": BCCz_lattice,
        "cubic": cubic_lattice,
        "FCC": FCC_lattice,
        "FBCC": FBCC_lattice,
        "isotruss": isotruss_lattice,
        "octet": octet_truss_lattice,
        "auxetic": auxetic_lattice,
    }

    for name, unit_cell in unit_cells.items():
        LatticeMicrostructure(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            L=L,
            radius=radius,
            unit_cell=unit_cell,
        ).write_h5(h5_filename, name)

    write_xdmf(
        h5_filepath=h5_filename,
        xdmf_filepath="data/lattice_microstructures.xdmf",
        microstructure_length=L[::-1],
        time_series=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
