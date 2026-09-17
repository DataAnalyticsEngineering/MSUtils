import numpy as np
from scipy.fft import irfftn, rfftn
from scipy.ndimage import median_filter as scipy_median_filter
from skimage.transform import resize


def _median_filter(image, footprint, workers):
    """Select the fastest periodic median filter for the image labels."""
    labels = np.unique(image)
    if labels.size == 1:
        return image
    if labels.size > 2:
        return scipy_median_filter(image, footprint=footprint, mode="wrap")

    kernel = np.zeros(image.shape, dtype=np.float32)
    positions = np.nonzero(footprint)
    radii = np.asarray(footprint.shape) // 2
    wrapped = tuple(
        (position - radius) % size
        for position, radius, size in zip(positions, radii, image.shape, strict=True)
    )
    np.add.at(kernel, wrapped, 1)

    kernel_spectrum = rfftn(kernel, workers=workers, overwrite_x=True)
    del kernel
    phase = (image == labels[1]).astype(np.float32)
    phase_spectrum = rfftn(phase, workers=workers, overwrite_x=True)
    del phase
    kernel_spectrum *= phase_spectrum
    del phase_spectrum
    counts = irfftn(kernel_spectrum, s=image.shape, workers=workers, overwrite_x=True)
    np.rint(counts, out=counts)
    return np.where(
        counts > np.count_nonzero(footprint) // 2, labels[1], labels[0]
    ).astype(image.dtype, copy=False)


def resize_image(data_array, target_resolution, workers=None):
    """Resize and periodically smooth a two- or three-dimensional label image."""
    image = np.asarray(data_array)
    if image.ndim not in (2, 3):
        raise ValueError("data_array must be two- or three-dimensional.")

    target = np.asarray(target_resolution, dtype=float)
    if (
        target.shape != (image.ndim,)
        or not np.all(np.isfinite(target))
        or not np.all(target == np.floor(target))
    ):
        raise ValueError("target_resolution must contain one integer per image axis.")
    target = target.astype(int)
    if np.any(target <= 0):
        raise ValueError("target_resolution values must be positive.")
    if np.array_equal(target, image.shape):
        return image.copy()

    lifted = resize(
        image,
        tuple(target),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(image.dtype, copy=False)
    if np.all(target <= image.shape):
        return lifted

    scale = np.ceil(target / np.asarray(image.shape)).astype(int)
    radii = 2 * scale
    coordinates = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
    footprint = (
        sum(
            np.abs(coordinate) / radius
            for coordinate, radius in zip(coordinates, radii, strict=True)
        )
        <= 1
    )
    return _median_filter(lifted, footprint, workers)
