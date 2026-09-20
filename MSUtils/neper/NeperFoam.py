from pathlib import Path

import numpy as np

from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.lattices.lattice_image import draw_strut
from MSUtils.neper.NeperMicrostructure import NeperMicrostructure


class NeperFoam(MicrostructureImage):
    """Rasterize the edges of a periodic Neper tessellation as struts."""

    def __init__(self, microstructure: NeperMicrostructure, strut_radius: float):
        vertices = []
        edges = set()
        with microstructure.tesr_filename.with_suffix(".obj").open() as file:
            for line in file:
                values = line.split()
                if values and values[0] == "v":
                    vertices.append(tuple(map(float, values[1:])))
                elif values and values[0] == "f":
                    face = [int(value) - 1 for value in values[1:]]
                    edges.update(
                        tuple(sorted(edge)) for edge in zip(face, face[1:] + face[:1])
                    )

        image = np.zeros(microstructure.shape, dtype=np.uint8)
        vertices = np.asarray(vertices)
        for start, end in edges:
            draw_strut(
                image,
                vertices[start],
                vertices[end],
                strut_radius,
                microstructure.voxel_size,
                microstructure.L,
                periodic=True,
            )

        self.strut_radius = float(strut_radius)
        super().__init__(
            image=image,
            grid=microstructure.grid,
            metadata={"strut_radius": self.strut_radius},
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
