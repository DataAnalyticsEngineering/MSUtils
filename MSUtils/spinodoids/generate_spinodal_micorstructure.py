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
    k0: Union[Tuple[float, float, float], List[float]] = (6.0, 6.0, 6.0),
    bandwidth: Union[Tuple[float, float, float], List[float]] = (0.6, 0.6, 0.6),
    exponent: float = 2.0,
    seed: int = 42,
    L: Union[Tuple[float, float, float], List[float]] = (1.0, 1.0, 1.0),
    rotation_matrix: Optional[np.ndarray] = np.eye(3, dtype=np.float32),
):
    """Generate a binary spinodal microstructure via spectral filtering.

    Parameters
    ----------
    shape : tuple or list
        Grid size (nz, ny, nx).
    volume_fraction : float, default 0.5
        Target volume fraction of the phase 1.
    k0 : array-like, default [6.0, 6.0, 6.0]
        Center wavenumber in cycles per domain (not angular) for each axis [k0_1, k0_2, k0_3].
    bandwidth : array-like, default [0.6, 0.6, 0.6]
        Relative Gaussian width (fraction of k0) for each axis [bw_1, bw_2, bw_3].
    exponent : float, default 2.0
        High-k rolloff exponent (larger -> stronger suppression of high frequencies).
    seed : int, default 42
        RNG seed for reproducibility.
    L : array-like, default [1.0, 1.0, 1.0]
    rotation_matrix : ndarray (ndim x ndim), default identity
        Orthonormal rotation matrix to apply to k-space coordinates.

    Returns
    -------
    image : ndarray (uint8) Binary microstructure.
    """
    np.random.seed(seed)
    ndim = len(shape)
    L = np.array(L, dtype=np.float32)
    dx_arr = L / np.array(shape, dtype=np.float32)
    k0_arr = np.array(k0, dtype=np.float32)
    bw_arr = np.array(bandwidth, dtype=np.float32)

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
        (grids[i] / ((np.float32(2.0 * np.pi) * k0_arr[i]) / L[i])) ** 2
        for i in range(ndim)
    )
    k = np.sqrt(k2_normalized)
    k0_ref = np.float32(1.0)  # Reference wavenumber is 1 after normalization

    del grids

    # Bandpass filter
    # Use mean bandwidth for the Gaussian envelope
    sigma = np.float32(np.mean(bw_arr) * k0_ref)
    k_diff = k - k0_ref
    filter_amp = np.exp(-0.5 * (k_diff / sigma) ** 2)

    # High-k rolloff (in-place multiplication)
    k_ratio = k / k0_ref
    filter_amp *= 1.0 / (1.0 + k_ratio**exponent)

    del k, k_diff, k_ratio

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
    k0 = [5.0, 20.0, 20.0]
    bandwidth = [0.1, 0.1, 0.1]
    exponent = 10.0
    seed = 42
    volume_fraction = 0.5
    euler_angles = [np.pi / 4, np.pi / 4, np.pi / 4]

    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    start_time = time()
    img = generate_spinodal_microstructure(
        N,
        volume_fraction=volume_fraction,
        k0=k0,
        bandwidth=bandwidth,
        exponent=exponent,
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
