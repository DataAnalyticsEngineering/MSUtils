from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import ConvexHull

_CORNERS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ],
    dtype=float,
)
_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
_EPS = np.finfo(float).eps
_NORMAL_ZERO_TOLERANCE = np.sqrt(_EPS)


def _lower_weighted_sum_cdf(value, widths):
    dimension = len(widths)
    if dimension == 1:
        return value / widths[0]
    if dimension == 2:
        a, b = widths
        result = value**2 - max(value - a, 0.0) ** 2 - max(value - b, 0.0) ** 2
        return min(max(result / (2.0 * a * b), 0.0), 1.0)

    a, b, c = widths
    result = (
        value**3
        - max(value - a, 0.0) ** 3
        - max(value - b, 0.0) ** 3
        - max(value - c, 0.0) ** 3
        + max(value - a - b, 0.0) ** 3
        + max(value - a - c, 0.0) ** 3
        + max(value - b - c, 0.0) ** 3
    )
    return min(max(result / (6.0 * a * b * c), 0.0), 1.0)


def _plane_offset(lower, spacing, normal, fraction_0):
    normal = np.array(normal, dtype=np.longdouble, copy=True)
    lower = np.asarray(lower, dtype=np.longdouble)
    spacing = np.asarray(spacing, dtype=np.longdouble)
    normal[np.abs(normal) < _NORMAL_ZERO_TOLERANCE * np.max(np.abs(normal))] = 0.0
    magnitude = np.linalg.norm(normal)
    if not np.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("Composite boxel normals must be finite and nonzero.")
    normal /= magnitude

    coefficients = normal * spacing
    widths = np.abs(coefficients)
    widths = widths[widths > _EPS * widths.max()]
    minimum = normal @ lower + coefficients[coefficients < 0.0].sum()
    target = 1.0 - fraction_0
    total = widths.sum()
    reflected = target > 0.5
    if reflected:
        target = 1.0 - target

    low = 0.0
    high = 0.5 * total
    for _ in range(44):
        middle = 0.5 * (low + high)
        if _lower_weighted_sum_cdf(middle, widths) < target:
            low = middle
        else:
            high = middle

    distance = 0.5 * (low + high)
    if reflected:
        distance = total - distance
    return np.asarray(normal, dtype=float), float(minimum + distance)


def _grid_points(shape, lengths):
    nx, ny, nz = shape
    hx, hy, hz = np.asarray(lengths) / shape
    points = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3))
    points[:, 0] = np.repeat(np.arange(nx + 1) * hx, (ny + 1) * (nz + 1))
    points[:, 1] = np.tile(np.repeat(np.arange(ny + 1) * hy, nz + 1), nx + 1)
    points[:, 2] = np.tile(np.arange(nz + 1) * hz, (nx + 1) * (ny + 1))
    return points


def _pure_hexahedra(coarse_image, index_dtype):
    _, ny, nz = coarse_image.shape
    pure = np.flatnonzero(coarse_image.ravel() < 2)
    i = pure // (ny * nz)
    j = pure // nz % ny
    k = pure % nz
    base = (i * (ny + 1) + j) * (nz + 1) + k
    x = (ny + 1) * (nz + 1)
    y = nz + 1
    offsets = np.array((0, x, x + y, y, 1, x + 1, x + y + 1, y + 1))
    return (base[:, None] + offsets).astype(index_dtype), coarse_image.ravel()[pure]


def _corner_ids(coords, shape, index_dtype):
    i, j, k = coords
    _, ny, nz = shape
    base = (i * (ny + 1) + j) * (nz + 1) + k
    x = (ny + 1) * (nz + 1)
    y = nz + 1
    return (base + np.array((0, x, x + y, y, 1, x + 1, x + y + 1, y + 1))).astype(
        index_dtype
    )


def _tetrahedralize(
    points, point_ids, extra_points, first_extra_id, face_cache, topology
):
    faces = face_cache.get(topology) if topology is not None else None
    if faces is None:
        faces = ConvexHull(points).simplices
        if topology is not None:
            face_cache[topology] = faces

    center = points.mean(axis=0)
    center_id = first_extra_id + len(extra_points)
    extra_points.append(center)

    tetrahedra = np.column_stack(
        (np.full(len(faces), center_id, dtype=point_ids.dtype), point_ids[faces])
    )
    determinants = np.linalg.det(points[faces] - center)
    negative = determinants < 0.0
    tetrahedra[negative, 2], tetrahedra[negative, 3] = (
        tetrahedra[negative, 3],
        tetrahedra[negative, 2].copy(),
    )
    return tetrahedra, np.abs(determinants).sum() / 6.0


def _composite_tetrahedra(
    info,
    shape,
    spacing,
    extra_points,
    first_extra_id,
    face_cache,
    index_dtype,
):
    fraction_0 = info.fraction_0
    lower = np.asarray(info.coords) * spacing
    normal, offset = _plane_offset(lower, spacing, info.normal, fraction_0)

    corners = lower + _CORNERS * spacing
    signed = corners @ normal - offset
    tolerance = 64 * _EPS * max(1.0, np.max(np.abs(corners)))
    signed[np.abs(signed) < tolerance] = 0.0

    intersections = []
    for first, second in _EDGES:
        if signed[first] * signed[second] < 0.0:
            position = signed[first] / (signed[first] - signed[second])
            intersections.append(
                corners[first] + position * (corners[second] - corners[first])
            )
    intersections = np.asarray(intersections).reshape(-1, 3)
    if np.count_nonzero(signed == 0.0) + len(intersections) < 3:
        raise RuntimeError(f"Could not intersect composite boxel {info.coords}.")

    intersection_ids = (
        first_extra_id
        + len(extra_points)
        + np.arange(len(intersections), dtype=index_dtype)
    )
    extra_points.extend(intersections)
    points = np.vstack((corners, intersections))
    point_ids = np.concatenate(
        (_corner_ids(info.coords, shape, index_dtype), intersection_ids)
    )
    intersection_indices = np.arange(8, len(points))

    tetrahedra = []
    boxel_volume = spacing.prod()
    cacheable = not np.any(signed == 0.0)
    for corner_mask, expected_fraction in (
        (signed >= 0.0, fraction_0),
        (signed <= 0.0, 1.0 - fraction_0),
    ):
        corner_indices = np.flatnonzero(corner_mask)
        indices = np.concatenate((corner_indices, intersection_indices))
        topology = sum(1 << index for index in corner_indices) if cacheable else None
        body_tetrahedra, volume = _tetrahedralize(
            points[indices],
            point_ids[indices],
            extra_points,
            first_extra_id,
            face_cache,
            topology,
        )
        error = abs(volume / boxel_volume - expected_fraction)
        if error > 1e-10 * (1.0 + abs(expected_fraction)):
            raise RuntimeError(f"Failed to preserve volume in boxel {info.coords}.")
        tetrahedra.append(body_tetrahedra)

    return tetrahedra


def create_combo_mesh(combo):
    """Create a ParaView-compatible volume mesh from a two-phase ComBo image."""
    if combo.coarse_image is None:
        raise ValueError("The ComBo image has not been downscaled.")
    coarse_image = np.asarray(combo.coarse_image)
    if coarse_image.ndim != 3:
        raise ValueError("coarse_image must be three-dimensional.")

    if not np.all(np.isin(coarse_image, (0, 1, 2))):
        raise ValueError("coarse_image must contain only labels 0, 1, and 2.")

    mixed_coords = {tuple(info.coords) for info in combo.voxel_info_list}
    if len(mixed_coords) != np.count_nonzero(coarse_image == 2) or any(
        coarse_image[coords] != 2 for coords in mixed_coords
    ):
        raise ValueError("Composite boxel metadata does not match coarse_image.")

    lengths = np.asarray(combo.lengths, dtype=float)
    spacing = lengths / coarse_image.shape
    maximum_points = np.prod(np.asarray(coarse_image.shape, dtype=np.int64) + 1)
    maximum_points += 8 * len(combo.voxel_info_list)
    index_dtype = np.int32 if maximum_points <= np.iinfo(np.int32).max else np.int64
    points = _grid_points(coarse_image.shape, lengths)
    hexahedra, hexahedron_materials = _pure_hexahedra(coarse_image, index_dtype)

    extra_points = []
    tetrahedra = []
    tetrahedron_materials = []
    face_cache = {}
    for info in combo.voxel_info_list:
        if not 0.0 < info.fraction_0 < 1.0:
            raise ValueError("Composite phase fractions must be between zero and one.")
        bodies = _composite_tetrahedra(
            info,
            coarse_image.shape,
            spacing,
            extra_points,
            len(points),
            face_cache,
            index_dtype,
        )
        for material, body in enumerate(bodies):
            tetrahedra.append(body)
            tetrahedron_materials.append(np.full(len(body), material, dtype=np.uint8))

    if extra_points:
        points = np.vstack((points, extra_points))

    cells = []
    material_data = []
    if len(hexahedra):
        cells.append(("hexahedron", hexahedra))
        material_data.append(np.asarray(hexahedron_materials, dtype=np.uint8))
    if tetrahedra:
        cells.append(("tetra", np.vstack(tetrahedra)))
        material_data.append(np.concatenate(tetrahedron_materials))

    return meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={"material_id": material_data},
    )


def write_combo_mesh(combo, filename):
    """Write a two-phase ComBo volume mesh as XDMF with HDF5 data."""
    path = Path(filename)
    if path.suffix.lower() not in {".xdmf", ".xmf"}:
        raise ValueError("filename must end in .xdmf or .xmf.")
    h5_path = path.with_suffix(".h5")
    meshio.write(path, create_combo_mesh(combo), file_format="xdmf", data_format="HDF")
    return path, h5_path
