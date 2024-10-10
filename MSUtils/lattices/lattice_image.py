import numpy as np
from lattice_definitions import *

from MSUtils.general.MicrostructureImage import MicrostructureImage


def physical_to_voxel(point, dimensions, shape):
    return np.round(point / dimensions * (np.array(shape) - 1)).astype(int)


def draw_strut(microstructure, start, end, radius, voxel_sizes, strut_type, L):
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    direction /= length

    num_points = int(length / np.linalg.norm(voxel_sizes))
    points = np.linspace(0, length, num_points)
    points = start + np.outer(points, direction)

    voxel_radius = [radius / voxel_sizes[i] for i in range(3)]

    for point in points:
        voxel_point = physical_to_voxel(point, L, microstructure.shape)
        x, y, z = voxel_point
        x_min, x_max = (
            max(0, int(x - voxel_radius[0])),
            min(microstructure.shape[0], int(x + voxel_radius[0] + 1)),
        )
        y_min, y_max = (
            max(0, int(y - voxel_radius[1])),
            min(microstructure.shape[1], int(y + voxel_radius[1] + 1)),
        )
        z_min, z_max = (
            max(0, int(z - voxel_radius[2])),
            min(microstructure.shape[2], int(z + voxel_radius[2] + 1)),
        )

        if strut_type == "circle":
            xx, yy, zz = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
            distance = (
                ((xx - x) * voxel_sizes[0]) ** 2
                + ((yy - y) * voxel_sizes[1]) ** 2
                + ((zz - z) * voxel_sizes[2]) ** 2
            )
            mask = distance <= radius**2

        microstructure[x_min:x_max, y_min:y_max, z_min:z_max][mask] = 1


def create_lattice_image(
    Nx, Ny, Nz, unit_cell_func, L=[1, 1, 1], radius=0.05, strut_type="circle"
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

    vertices, edges = unit_cell_func()
    voxel_sizes = [L[i] / [Nx, Ny, Nz][i] for i in range(3)]

    microstructure = np.zeros((Nx, Ny, Nz), dtype=np.int8)
    for edge in edges:
        start, end = vertices[edge[0]] * L, vertices[edge[1]] * L
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

    from MSUtils.general.h52xdmf import write_xdmf

    write_xdmf(
        "data/lattice_microstructures.h5",
        "data/lattice_microstructures.xdmf",
        L,
        False,
        None,
        True,
    )

    # vertices, edges = octet_truss_lattice()
    # plot_lattice(vertices, edges)
    # print(check_rigidity(vertices, edges))
