from time import time
import numpy as np
from typing import Tuple, List, Union, Optional
from scipy.fft import fftn, ifftn, fftfreq
from scipy.spatial.transform import Rotation
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.general.h52xdmf import write_xdmf


def generate_spinodal_microstructure(
    shape: Union[Tuple[int, int, int], List[int]],
    volume_fraction: float = 0.5,
    wavenumber: Union[Tuple[float, float, float], List[float]] = (6.0, 6.0, 6.0),
    sigma: float = 0.5,
    seed: int = 42,
    L: Union[Tuple[float, float, float], List[float]] = (1.0, 1.0, 1.0),
    rotation_matrix: Optional[np.ndarray] = None,
):
    """Generate a binary spinodal microstructure via spectral filtering.

    Parameters
    ----------
    shape : tuple or list
        Grid size (nz, ny, nx).
    volume_fraction : float, default 0.5
        Target volume fraction of phase 1.
    wavenumber : array-like, default [6.0, 6.0, 6.0]
        Center wavenumber in cycles per domain for each axis [k1, k2, k3].
    sigma : float, default 0.5
        Standard deviation of Gaussian bandpass filter (in normalized k-space).
    seed : int, default 42
        Random seed for reproducibility.
    L : array-like, default [1.0, 1.0, 1.0]
        Physical domain dimensions [Lx, Ly, Lz].
    rotation_matrix : ndarray (3x3), optional
        Orthonormal rotation matrix for orienting anisotropic features.
        If None, uses identity (no rotation).

    Returns
    -------
    image : ndarray (uint8)
        Binary microstructure with values 0 and 1.
    """
    np.random.seed(seed)
    ndim = len(shape)
    L = np.array(L, dtype=np.float32)
    dx_arr = L / np.array(shape, dtype=np.float32)
    wavenumber_arr = np.array(wavenumber, dtype=np.float32)
    if rotation_matrix is None:
        rotation_matrix = np.eye(ndim, dtype=np.float32)

    # Generate noise (resolution dependent)
    noise = np.random.randn(*shape).astype(np.float32, copy=False)
    noise_hat = fftn(noise, workers=-1, overwrite_x=True)

    # Build k-space grids
    grids = [
        fftfreq(shape[i], d=dx_arr[i]).astype(np.float32) * (2.0 * np.pi)
        for i in range(ndim)
    ]
    grids = np.meshgrid(*grids, indexing="ij")

    # Rotate k-space coordinates: k_rotated = R @ k_original
    R = np.array(rotation_matrix, dtype=np.float32)
    grids = list(np.einsum("ij,jklm->iklm", R, np.array(grids)))

    # Compute effective radial wavenumber using ellipsoidal normalization
    k2_normalized = sum(
        (grids[i] / ((np.float32(2.0 * np.pi) * wavenumber_arr[i]) / L[i])) ** 2
        for i in range(ndim)
    )
    k = np.sqrt(k2_normalized)

    del grids

    # Gaussian bandpass filter centered at k=1 (in normalized space)
    k_diff = k - np.float32(1.0)
    filter_amp = np.exp(-0.5 * (k_diff / np.float32(sigma)) ** 2)

    del k, k_diff

    # Apply filter in-place
    noise_hat *= filter_amp
    del filter_amp

    # Inverse FFT
    field = np.real(ifftn(noise_hat, workers=-1, overwrite_x=True))

    field = (field - field.mean()) / field.std()

    # Threshold
    thr = np.quantile(field, 1.0 - volume_fraction)
    image = (field >= thr).astype(np.uint8)
    return image


if __name__ == "__main__":

    N = [256, 256, 256]
    L = [1.0, 1.0, 1.0]
    wavenumber = [20.0, 20.0, 20.0]
    sigma = 0.01
    seed = 42
    volume_fraction = 0.5
    euler_angles = [0, 0, 0]

    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    start_time = time()
    img = generate_spinodal_microstructure(
        N,
        volume_fraction=volume_fraction,
        wavenumber=wavenumber,
        sigma=sigma,
        seed=seed,
        L=L,
        rotation_matrix=R,
    )
    end_time = time()
    print(
        f"3D spinodal microstructure generation took {end_time - start_time:.4f} seconds"
    )

    MS = MicrostructureImage(image=img, L=L)
    MS.write(
        h5_filename="data/spinodoids.h5",
        dset_name="/dset_0",
        order="zyx",
    )
    write_xdmf(
        h5_filepath="data/spinodoids.h5",
        xdmf_filepath="data/spinodoids.xdmf",
        microstructure_length=L[::-1],
    )
