import numpy as np


def draw_strut(image, start, end, radius, grid, *, periodic=False, value=1):
    """Rasterize a capsule (cylinder with hemispherical end caps) into `image`, in place.

    `start`, `end` and `radius` are in physical units on the domain [0, L)^3; voxel i
    along each axis has its center at (i + 0.5) * h. With `periodic=True` the capsule
    wraps around the domain boundaries.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = np.asarray(image.shape)
    h = np.asarray(grid.lengths, float) / n
    axis = end - start
    if not axis.any():
        return
    inv_len2 = 1.0 / (axis @ axis)

    # Voxel index range of the bounding box, rounded outward so ties are never cut off.
    lo = np.floor((np.minimum(start, end) - radius) / h - 0.5).astype(int)
    hi = np.ceil((np.maximum(start, end) + radius) / h - 0.5).astype(int) + 1
    if not periodic:
        lo, hi = np.maximum(lo, 0), np.minimum(hi, n)
        if np.any(lo >= hi):
            return
    ix, iy, iz = (np.arange(a, b) for a, b in zip(lo, hi))

    # Process the box in x-slabs to bound memory.
    step = max(1, 1_000_000 // (len(iy) * len(iz)))
    for x0 in range(0, len(ix), step):
        idx = (ix[x0 : x0 + step], iy, iz)
        d = np.ix_(*[(i + 0.5) * hk - sk for i, hk, sk in zip(idx, h, start)])
        t = np.clip(sum(dk * ak for dk, ak in zip(d, axis)) * inv_len2, 0.0, 1.0)
        dist2 = sum((dk - t * ak) ** 2 for dk, ak in zip(d, axis))
        inside = np.nonzero(dist2 <= radius**2)
        # Periodic images are handled by wrapping indices (a no-op when not periodic).
        image[tuple(i[k] % nk for i, k, nk in zip(idx, inside, n))] = value
