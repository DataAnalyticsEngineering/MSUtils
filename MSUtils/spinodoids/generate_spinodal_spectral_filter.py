from time import time
import numpy as np
from typing import Tuple, List, Union
from scipy.fft import fftn, ifftn, fftfreq
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
    anisotropy_axes: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]] = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ),
):
    """Generate a binary spinodal microstructure via spectral filtering.

    Parameters
    ----------
    shape : tuple or list
        Grid size (nz, ny, nx).
    k0 : array-like, default [6.0, 6.0, 6.0]
        Center wavenumber in cycles per domain (not angular) for each axis [k0_v3, k0_v2, k0_v1].
    bandwidth : array-like, default [0.6, 0.6, 0.6]
        Relative Gaussian width (fraction of k0) for each axis [bw_v3, bw_v2, bw_v1].
        Use same values for isotropic bandwidth, different values for anisotropic.
    exponent : float
        High-k rolloff exponent (larger -> stronger suppression)
    seed : int, optional
        RNG seed for reproducibility
    L : array-like, default [1.0, 1.0, 1.0]
        Physical dimensions [Lz, Ly, Lx].
    anisotropy_axes : 2-tuple or list of vectors, default [(1,0,0), (0,1,0)]
        Define a preferred coordinate system (v1, v2) for oblique orientations.
        v1, v2 should be orthogonal unit vectors defining preferred directions.
    """
    np.random.seed(seed)
    ndim = len(shape)
    L = np.array(L, dtype=np.float32)
    dx_arr = L / np.array(shape, dtype=np.float32)
    k0_arr = np.array(k0, dtype=np.float32)
    bw_arr = np.array(bandwidth, dtype=np.float32)

    np.random.seed(seed)
    noise = np.random.randn(*shape).astype(
        np.float32, copy=False
    )  # Generate noise (resolution dependent)
    noise_hat = fftn(noise, workers=-1, overwrite_x=True)

    # Build k-space grids
    grids = [
        fftfreq(shape[i], d=dx_arr[i]).astype(np.float32) * (2.0 * np.pi)
        for i in range(ndim)
    ]
    grids = np.meshgrid(*grids, indexing="ij")

    # Build rotation matrix: R = [v1, v2, v3]^T
    v1, v2 = anisotropy_axes
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    v3 = np.cross(v1, v2)
    R = np.array([v1, v2, v3], dtype=np.float32)

    # Rotate k-space coordinates: k_rotated = R @ k_original
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

    v1 = np.array([1.0, 1.0, 1.0])
    v1 /= np.linalg.norm(v1)
    v2 = np.array([1.0, -1.0, 0.0])
    v2 /= np.linalg.norm(v2)
    anisotropy_axes = (v1, v2)

    start_time = time()
    img = generate_spinodal_microstructure(
        N,
        volume_fraction=volume_fraction,
        k0=k0,
        bandwidth=bandwidth,
        exponent=exponent,
        seed=seed,
        L=L,
    )
    end_time = time()
    print(
        f"3D spinodal microstructure generation took {end_time - start_time:.4f} seconds"
    )

    MS = MicrostructureImage(image=img, L=L)
    MS.write(
        h5_filename="data/spinodal_spectral.h5",
        dset_name="/spinodal_spectral",
        order="zyx",
    )
    write_xdmf(
        h5_filepath="data/spinodal_spectral.h5",
        xdmf_filepath="data/spinodal_spectral.xdmf",
        microstructure_length=L[::-1],
    )
