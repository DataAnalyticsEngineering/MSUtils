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


def physical_to_voxel(point, dimensions, shape):
    """
    Map a physical coordinate in [0, L] to voxel index in [0, N-1],
    consistently with spacing = L/(N-1).
    """
    point = np.asarray(point, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.int64)
    dimensions = np.asarray(dimensions, dtype=np.float64)

    # Map [0, L] -> [0, N-1], then clamp
    idx = np.floor(point / dimensions * (shape - 1) + 0.5).astype(np.int64)
    idx = np.clip(idx, 0, shape - 1)
    return idx


def draw_strut(microstructure, start, end, radius, voxel_sizes, strut_type, L):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return
    direction /= length

    # Step roughly one voxel diagonal in PHYSICAL space
    step = np.linalg.norm(voxel_sizes)
    num_points = max(1, int(np.ceil(length / step)))
    ts = np.linspace(0.0, length, num_points, dtype=np.float64)
    points = start + np.outer(ts, direction)

    # Radius window in index units for bounding box
    voxel_radius = np.array(
        [radius / voxel_sizes[i] for i in range(3)], dtype=np.float64
    )

    for p in points:
        x, y, z = physical_to_voxel(p, L, microstructure.shape)

        x_min = max(0, int(np.floor(x - voxel_radius[0])))
        x_max = min(microstructure.shape[0], int(np.ceil(x + voxel_radius[0] + 1)))
        y_min = max(0, int(np.floor(y - voxel_radius[1])))
        y_max = min(microstructure.shape[1], int(np.ceil(y + voxel_radius[1] + 1)))
        z_min = max(0, int(np.floor(z - voxel_radius[2])))
        z_max = min(microstructure.shape[2], int(np.ceil(z + voxel_radius[2] + 1)))

        if strut_type == "circle":
            xx, yy, zz = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
            # Compute PHYSICAL distance (consistent spacing)
            dx = (xx - x) * voxel_sizes[0]
            dy = (yy - y) * voxel_sizes[1]
            dz = (zz - z) * voxel_sizes[2]
            distance2 = dx * dx + dy * dy + dz * dz
            mask = distance2 <= (radius * radius)
            microstructure[x_min:x_max, y_min:y_max, z_min:z_max][mask] = 1


def create_lattice_image(
    Nx, Ny, Nz, unit_cell_func, L=None, radius=0.05, strut_type="circle"
):
    """
    Create a lattice microstructure image.

    Parameters:
    - Nx, Ny, Nz: int - The resolution of the microstructure in each dimension.
    - unit_cell_func: function - The function that returns the vertices and edges of the unit cell.
    - L: list - The length of the microstructure in each dimension. Default is [1, 1, 1].
    - radius: float - The radius of the struts. Default is 0.05.
    - strut_type: str - The type of the struts. Default is 'circle'.

    Returns:
    - microstructure: ndarray - The generated microstructure image.
    """
    if L is None:
        L = [1.0, 1.0, 1.0]
    L = np.asarray(L, dtype=np.float64)

    vertices, edges = unit_cell_func()

    # CONSISTENT voxel spacing with 0..N-1 indexing
    Nvec = np.array([Nx, Ny, Nz], dtype=np.int64)
    voxel_sizes = L / (Nvec - 1).astype(np.float64)

    microstructure = np.zeros((Nx, Ny, Nz), dtype=np.int8)

    # scale vertices into physical domain [0, L]
    phys_vertices = vertices * L

    for a, b in edges:
        start, end = phys_vertices[a], phys_vertices[b]
        draw_strut(microstructure, start, end, radius, voxel_sizes, strut_type, L)

    return microstructure


if __name__ == "__main__":
    Nx, Ny, Nz = 256, 256, 256  # microstructure resolution
    L = [1.0, 1.0, 1.0]  # microstructure length
    radius = 0.05  # radius of the struts
    strut_type = "circle"

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
        "strut type": strut_type,
    }

    microstructures = {}
    for name, unit_cell_func in unit_cell_types.items():
        image = create_lattice_image(Nx, Ny, Nz, unit_cell_func, L, radius, strut_type)

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

    # vertices, edges = auxetic_lattice()
    # plot_lattice(vertices, edges)
    # print(check_rigidity(vertices, edges))
