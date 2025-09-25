import numpy as np
from MSUtils.lattices.lattice_image import draw_strut, _inside_cell
from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation
from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.general.h52xdmf import write_xdmf


def _all_voronoi_edges(tess):
    """
    Extract all edges from a PeriodicVoronoiTessellation object.
    Each edge is represented as a tuple of its two endpoints.
    """
    vor = tess.voronoi
    edges = []
    for verts in vor.ridge_vertices:
        if any(v == -1 for v in verts):
            continue
        if len(verts) < 2:
            continue
        cyc = list(verts) + [verts[0]]
        for u, v in zip(cyc[:-1], cyc[1:]):
            edges.append((vor.vertices[u].astype(float), vor.vertices[v].astype(float)))
    return edges


def periodic_voronoi_foam(
    tess, N, L, strut_radius, dtype=np.uint8, strut_type="circle"
):
    """
    Use FULL Voronoi (with duplicated seeds) and draw every edge whose
    start or end lies inside [0,L)^d. No periodic translations needed.
    """
    N = np.asarray(N, int)
    L = np.asarray(L, float)

    micro = np.zeros(tuple(N.tolist()), dtype=dtype)
    voxel_sizes = L / (N - 1).astype(float)

    edges = _all_voronoi_edges(tess)

    # De-dup to avoid overdrawing: hash by midpoint & direction (sign-agnostic)
    seen = set()

    for V0, V1 in edges:
        if not (_inside_cell(V0, L) or _inside_cell(V1, L)):
            continue

        mid = 0.5 * (V0 + V1)
        d = V1 - V0
        nrm = np.linalg.norm(d) + 1e-15
        diru = np.round(d / nrm, 6)
        key = tuple(np.round(mid / np.maximum(L, 1.0), 6)) + tuple(
            np.minimum(diru, -diru)
        )
        if key in seen:
            continue
        seen.add(key)

        draw_strut(micro, V0, V1, strut_radius, voxel_sizes, strut_type, L)

    return micro


if __name__ == "__main__":

    num_crystals = 125
    L = [1, 1, 1]
    Nx, Ny, Nz = 512, 512, 512
    permute_order = "zyx"
    strut_radius = 0.01

    # Generate Voronoi seeds and tessellation
    SeedInfo = VoronoiSeeds(num_crystals, L, "sobol", BitGeneratorSeed=42)
    tess = PeriodicVoronoiTessellation(L, SeedInfo.seeds)

    # Generate Voronoi foam image
    micro = periodic_voronoi_foam(
        tess, [Nx, Ny, Nz], L, strut_radius, strut_type="circle"
    )

    IMG = MicrostructureImage(image=micro, L=L)
    IMG.write("data/voro_foam.h5", "/dset_0", order=permute_order)

    write_xdmf(
        "data/voro_foam.h5",
        "data/voro_foam.xdmf",
        microstructure_length=L[::-1],
    )
