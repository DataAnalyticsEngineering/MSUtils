import numpy as np


def interface_laplacian(image, lengths=(1.0, 1.0, 1.0)):
    """Compute the absolute periodic Laplacian of an image."""
    image = np.asarray(image, dtype=float)
    voxel_sizes = np.asarray(lengths, dtype=float) / image.shape
    laplacian = np.zeros(image.shape)
    for axis, spacing in enumerate(voxel_sizes):
        laplacian += (
            np.roll(image, 1, axis) - 2.0 * image + np.roll(image, -1, axis)
        ) / spacing**2
    return np.abs(laplacian), voxel_sizes


def _phase_centroid(image, voxel_sizes):
    axes = [
        (np.arange(size) + 0.5 - 0.5 * size) * spacing
        for size, spacing in zip(image.shape, voxel_sizes, strict=True)
    ]
    return np.array(
        [
            image.sum(axis=(1, 2)) @ axes[0],
            image.sum(axis=(0, 2)) @ axes[1],
            image.sum(axis=(0, 1)) @ axes[2],
        ]
    )


def c2c_normal(image, voxel_sizes):
    """Estimate a normal from the phase-centroid direction."""
    normal = -_phase_centroid(image, voxel_sizes)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0.0:
        raise ValueError("The C2C normal is undefined for coincident phase centroids.")
    return normal / magnitude


def combo_normal(image, interface_weights, voxel_sizes):
    """Estimate a normal from a weighted interface-plane fit."""
    weights = np.abs(interface_weights)
    interface = np.where(weights > 1e-5)
    weights = weights[interface].flatten()
    weights /= weights.sum()

    coordinates = voxel_sizes[:, None] * np.array(interface)
    coordinates -= (coordinates * weights).sum(axis=1)[:, None]
    moment = coordinates @ (coordinates * weights).T
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    normal /= np.linalg.norm(normal)

    if _phase_centroid(image, voxel_sizes) @ normal > 0.0:
        normal *= -1.0

    return normal
