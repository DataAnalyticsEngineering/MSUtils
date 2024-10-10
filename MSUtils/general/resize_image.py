import numpy as np
import scipy.ndimage
import skimage.morphology
import skimage.transform


def resize_image(data_array, scale=None, target_resolution=None):
    """
    resizes a 3D data array by a given scale factor or to a target resolution.

    Parameters:
    - data_array: numpy.ndarray
        The input 3D data array to be resized.
    - scale: list, optional
        The scale factor to be applied along each dimension. If provided, 'target_resolution' is ignored.
    - target_resolution: list, optional
        The target resolution of the resized array. Used only if 'scale' is not provided.

    Returns:
    - resized_image: numpy.ndarray
        The resized and smoothed 3D data array.
    """
    if scale is not None:
        new_shape = np.multiply(data_array.shape, scale)
    elif target_resolution is not None:
        scale = np.ceil(
            np.array(target_resolution) / np.array(data_array.shape)
        ).astype(int)
        new_shape = target_resolution
    else:
        scale = [2, 2, 2]
        new_shape = np.multiply(data_array.shape, scale)

    resized_image = skimage.transform.resize(
        data_array, new_shape, preserve_range=True, anti_aliasing=False, order=0
    )

    if np.all(scale > [1, 1, 1]):
        if np.all(scale[0] == scale[1] == scale[2]):
            radius = int(scale[0] * 2)  # Adjust the radius as needed
            footprint = skimage.morphology.octahedron(radius)
            resized_image = scipy.ndimage.median_filter(
                resized_image, footprint=footprint, mode="wrap"
            )
        else:
            kernel_size = [
                2 * s for s in scale
            ]  # Default kernel size is twice the scale factor
            resized_image = scipy.ndimage.median_filter(
                resized_image, size=kernel_size, mode="wrap"
            )

    return resized_image


if __name__ == "__main__":
    from MicrostructureImage import MicrostructureImage

    ms = MicrostructureImage(h5_filename="data/fibers1.h5", dset_name="/img")
    ms_resized = MicrostructureImage(
        image=resize_image(ms.image, target_resolution=[512, 512, 512])
    )
    ms_resized.write(h5_filename="data/test_resize.h5", dset_name="ms")

    error = {}
    for key in ms.volume_fractions.keys():
        error[key] = (
            (ms.volume_fractions[key] - ms_resized.volume_fractions[key])
            * 100
            / ms.volume_fractions[key]
        )
        print(f"Resizing volume fraction error for phase {key}: {error[key]:.6f}%")
