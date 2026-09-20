from itertools import product

import numpy as np

from MSUtils.general.h52xdmf import write_xdmf
from MSUtils.general.MicrostructureImage import MicrostructureImage
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


def draw_strut(
    microstructure,
    start,
    end,
    radius,
    voxel_sizes,
    L,
    *,
    periodic=False,
):
    """Rasterize a capsule, optionally wrapping it across the image domain."""
    if microstructure.ndim != 3:
        raise ValueError("microstructure must be three-dimensional.")

    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    voxel_sizes = np.asarray(voxel_sizes, np.float64)
    L = np.asarray(L, np.float64)
    shape = np.asarray(microstructure.shape)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must contain three coordinates.")
    if voxel_sizes.shape != (3,) or L.shape != (3,):
        raise ValueError("voxel_sizes and L must contain three values.")
    if not np.all(np.isfinite((*start, *end, *voxel_sizes, *L, radius))):
        raise ValueError("Strut geometry must contain only finite values.")
    if radius <= 0 or np.any(voxel_sizes <= 0) or np.any(L <= 0):
        raise ValueError("radius, voxel_sizes, and L must be positive.")
    if not np.allclose(voxel_sizes * shape, L):
        raise ValueError("voxel_sizes must equal L divided by the image shape.")

    segment = end - start
    length_squared = segment @ segment
    if length_squared == 0:
        return

    eps = 10 * np.finfo(np.float64).eps
    shifts = ((0, 0, 0),)
    if periodic:
        center_min = 0.5 * voxel_sizes
        center_max = L - center_min
        segment_min = np.minimum(start, end)
        segment_max = np.maximum(start, end)
        first_shift = np.ceil((center_min - segment_max - radius) / L - eps).astype(int)
        last_shift = np.floor((center_max - segment_min + radius) / L + eps).astype(int)
        shifts = product(
            *(
                range(first, last + 1)
                for first, last in zip(first_shift, last_shift, strict=True)
            )
        )

    radius_squared = radius * radius
    for shift in shifts:
        shifted_start = start + np.asarray(shift) * L
        shifted_end = shifted_start + segment
        lower = np.minimum(shifted_start, shifted_end) - radius
        upper = np.maximum(shifted_start, shifted_end) + radius
        first = np.maximum(0, np.ceil(lower / voxel_sizes - 0.5 - eps).astype(int))
        last = np.minimum(
            shape, np.floor(upper / voxel_sizes - 0.5 + eps).astype(int) + 1
        )
        if np.any(first >= last):
            continue

        y = ((np.arange(first[1], last[1]) + 0.5) * voxel_sizes[1])[None, :, None]
        z = ((np.arange(first[2], last[2]) + 0.5) * voxel_sizes[2])[None, None, :]
        plane_size = (last[1] - first[1]) * (last[2] - first[2])
        chunk_size = max(1, 1_000_000 // plane_size)

        for x_start in range(first[0], last[0], chunk_size):
            x_stop = min(x_start + chunk_size, last[0])
            x = ((np.arange(x_start, x_stop) + 0.5) * voxel_sizes[0])[:, None, None]
            projection = (
                (x - shifted_start[0]) * segment[0]
                + (y - shifted_start[1]) * segment[1]
                + (z - shifted_start[2]) * segment[2]
            ) / length_squared
            np.clip(projection, 0.0, 1.0, out=projection)
            distance_squared = (
                (x - shifted_start[0] - projection * segment[0]) ** 2
                + (y - shifted_start[1] - projection * segment[1]) ** 2
                + (z - shifted_start[2] - projection * segment[2]) ** 2
            )
            block = microstructure[
                x_start:x_stop,
                first[1] : last[1],
                first[2] : last[2],
            ]
            block[distance_squared <= radius_squared] = 1


def create_lattice_image(Nx, Ny, Nz, unit_cell_func, L=None, radius=0.05):
    """
    Create a lattice microstructure image.

    Parameters:
    - Nx, Ny, Nz: int - The resolution of the microstructure in each dimension.
    - unit_cell_func: function - The function that returns the vertices and edges of the unit cell.
    - L: list - The length of the microstructure in each dimension. Default is [1, 1, 1].
    - radius: float - The radius of the struts. Default is 0.05.

    Returns:
    - microstructure: ndarray - The generated microstructure image.
    """
    resolution = np.asarray((Nx, Ny, Nz), dtype=np.float64)
    if not np.all(np.isfinite(resolution)) or not np.all(
        resolution == np.floor(resolution)
    ):
        raise ValueError("Resolution values must be integers.")
    Nvec = resolution.astype(np.int64)
    L = np.ones(3) if L is None else np.asarray(L, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("L must contain three values.")
    if np.any(Nvec <= 0) or np.any(~np.isfinite(L)) or np.any(L <= 0):
        raise ValueError("Resolution and physical lengths must be positive.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive.")
    vertices, edges = unit_cell_func()
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Unit-cell vertices must have shape (n, 3).")
    if np.any(~np.isfinite(vertices)):
        raise ValueError("Unit-cell vertices must be finite.")
    edge_values = np.asarray(edges, dtype=np.float64)
    if edge_values.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    else:
        if (
            edge_values.ndim != 2
            or edge_values.shape[1] != 2
            or np.any(~np.isfinite(edge_values))
            or np.any(edge_values != np.floor(edge_values))
        ):
            raise ValueError("Unit-cell edges must contain pairs of vertex indices.")
        edges = edge_values.astype(np.int64)
        if np.any(edges < 0) or np.any(edges >= len(vertices)):
            raise ValueError("Unit-cell edge index is out of bounds.")
        if np.any(np.all(vertices[edges[:, 0]] == vertices[edges[:, 1]], axis=1)):
            raise ValueError("Unit-cell edges must have nonzero length.")

    voxel_sizes = L / Nvec

    microstructure = np.zeros(tuple(Nvec), dtype=np.int8)

    # scale vertices into physical domain [0, L]
    phys_vertices = vertices * L

    for a, b in edges:
        start, end = phys_vertices[a], phys_vertices[b]
        draw_strut(
            microstructure,
            start,
            end,
            radius,
            voxel_sizes,
            L,
            periodic=True,
        )

    return microstructure


if __name__ == "__main__":
    Nx, Ny, Nz = 200, 300, 400  # microstructure resolution
    L = [1.0, 1.0, 1.0]  # microstructure length
    radius = 0.05  # radius of the struts

    unit_cell_types = {
        "BCC": BCC_lattice,
        "BCCz": BCCz_lattice,
        "cubic": cubic_lattice,
        "FCC": FCC_lattice,
        "FBCC": FBCC_lattice,
        "isotruss": isotruss_lattice,
        "octet": octet_truss_lattice,
        "auxetic": auxetic_lattice,
        # Add more unit cells here...
    }

    metadata = {
        "resolution [Nx, Ny, Nz]": [Nx, Ny, Nz],
        "length [Lx, Ly, Lz]": L,
        "strut radius": radius,
    }

    microstructures = {}
    for name, unit_cell_func in unit_cell_types.items():
        image = create_lattice_image(Nx, Ny, Nz, unit_cell_func, L, radius)

        tmp_metadata = metadata.copy()
        tmp_metadata["lattice type"] = name
        microstructures[name] = MicrostructureImage(image=image, metadata=tmp_metadata)
        microstructures[name].write(
            h5_filename="data/lattice_microstructures.h5", dset_name=name
        )

    write_xdmf(
        h5_filepath="data/lattice_microstructures.h5",
        xdmf_filepath="data/lattice_microstructures.xdmf",
        microstructure_length=L[::-1],
        time_series=False,
        verbose=True,
    )
