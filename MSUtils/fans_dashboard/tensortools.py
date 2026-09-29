import numpy as np
from scipy.linalg import eigvalsh

"""
Voigt strain: (ε_11, ε_22, ε_33, 2ε_23, 2ε_13, 2ε_12).
Voigt stress: (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12).

Mandel: (A_11, A_22, A_33, √2 A_12, √2 A_13, √2 A_23).
"""

_COMPONENT_ORDER = [0, 1, 2, 5, 4, 3]
_SQRT2 = np.sqrt(2.0)


def _convert_vector(values, shear_factor, order):
    result = np.array(values, dtype=float)
    result[..., 3:] *= shear_factor
    return result[..., _COMPONENT_ORDER] if order == "voigt" else result


def _convert_matrix(values, shear_factor, order):
    result = np.array(values, dtype=float)
    result[..., 3:, :] *= shear_factor
    result[..., :, 3:] *= shear_factor
    if order == "voigt":
        result = result[..., _COMPONENT_ORDER, :][..., :, _COMPONENT_ORDER]
    return result


def VoigtStrain2Mandel(A_voigt, order="voigt"):
    """Convert engineering Voigt strain to Mandel notation."""
    return _convert_vector(A_voigt, 1.0 / _SQRT2, order)


def VoigtStress2Mandel(A_voigt, order="voigt"):
    """Convert Voigt stress to Mandel notation."""
    return _convert_vector(A_voigt, _SQRT2, order)


def Mandel2VoigtStrain(A_mandel, order="voigt"):
    """Convert Mandel strain to engineering Voigt notation."""
    return _convert_vector(A_mandel, _SQRT2, order)


def Mandel2VoigtStress(A_mandel, order="voigt"):
    """Convert Mandel stress to Voigt notation."""
    return _convert_vector(A_mandel, 1.0 / _SQRT2, order)


def StiffnessVoigt2Mandel(C_v, order="voigt"):
    """Convert a Voigt stiffness matrix to Mandel notation."""
    return _convert_matrix(C_v, _SQRT2, order)


def ComplianceVoigt2Mandel(S_v, order="voigt"):
    """Convert a Voigt compliance matrix to Mandel notation."""
    return _convert_matrix(S_v, 1.0 / _SQRT2, order)


def Full2Mandel(A):
    """Convert symmetric 3x3 tensors to Mandel notation."""
    A = np.asarray(A, dtype=float)
    A_mandel = np.empty(A.shape[:-2] + (6,))
    A_mandel[..., 0] = A[..., 0, 0]
    A_mandel[..., 1] = A[..., 1, 1]
    A_mandel[..., 2] = A[..., 2, 2]
    A_mandel[..., 3] = _SQRT2 * A[..., 0, 1]
    A_mandel[..., 4] = _SQRT2 * A[..., 0, 2]
    A_mandel[..., 5] = _SQRT2 * A[..., 1, 2]
    return A_mandel


def Mandel2Full(A_mandel):
    """Convert Mandel vectors to symmetric 3x3 tensors."""
    A_mandel = np.asarray(A_mandel, dtype=float)
    A = np.empty(A_mandel.shape[:-1] + (3, 3))
    A[..., 0, 0] = A_mandel[..., 0]
    A[..., 1, 1] = A_mandel[..., 1]
    A[..., 2, 2] = A_mandel[..., 2]
    A[..., 0, 1] = A_mandel[..., 3] / _SQRT2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 0, 2] = A_mandel[..., 4] / _SQRT2
    A[..., 2, 0] = A[..., 0, 2]
    A[..., 1, 2] = A_mandel[..., 5] / _SQRT2
    A[..., 2, 1] = A[..., 1, 2]
    return A


def IsoProjectionKappa(A_mandel):
    """Project Mandel tensors onto the second-order identity."""
    A_mandel = np.asarray(A_mandel)
    return A_mandel[..., :3].mean(axis=-1)


def IsoProjectionC(C_mandel):
    """Return the isotropic bulk and shear projections of Mandel matrices."""
    C_mandel = np.asarray(C_mandel)
    K = C_mandel[..., :3, :3].mean(axis=(-2, -1))
    G = (np.trace(C_mandel, axis1=-2, axis2=-1) - 3.0 * K) / 10.0
    return K, G


def Piso1():
    """Return the volumetric isotropic projector in Mandel notation."""
    P = np.zeros((6, 6))
    P[:3, :3] = 1.0 / 3.0
    return P


def Piso2():
    """Return the deviatoric isotropic projector in Mandel notation."""
    return np.eye(6) - Piso1()


def Ciso(K, G):
    """Return isotropic Mandel stiffness matrices from bulk and shear moduli."""
    K = np.asarray(K, dtype=float)[..., None, None]
    G = np.asarray(G, dtype=float)[..., None, None]
    return 3.0 * K * Piso1() + 2.0 * G * Piso2()


def ConvertElasticConstants(**kwargs):
    """Return all isotropic elastic constants from exactly two inputs."""
    names = {"E", "nu", "G", "K"}
    unknown = set(kwargs) - names
    if unknown:
        raise ValueError(f"Unknown elastic constants: {', '.join(sorted(unknown))}.")

    values = {name: float(value) for name, value in kwargs.items() if value is not None}
    if len(values) != 2:
        raise ValueError("Exactly two of E, nu, G, and K must be provided.")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Elastic constants must be finite.")
    for name in ("E", "G", "K"):
        if name in values and values[name] <= 0.0:
            raise ValueError("E, G, and K must be positive.")
    if "nu" in values and not -1.0 < values["nu"] < 0.5:
        raise ValueError("nu must satisfy -1 < nu < 0.5.")

    E = values.get("E")
    nu = values.get("nu")
    G = values.get("G")
    K = values.get("K")
    pair = set(values)

    if pair == {"E", "nu"}:
        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
    elif pair == {"E", "G"}:
        if E >= 3.0 * G:
            raise ValueError("E and G are not physically consistent.")
        nu = E / (2.0 * G) - 1.0
        K = E / (3.0 * (1.0 - 2.0 * nu))
    elif pair == {"E", "K"}:
        if E >= 9.0 * K:
            raise ValueError("E and K are not physically consistent.")
        nu = (3.0 * K - E) / (6.0 * K)
        G = E / (2.0 * (1.0 + nu))
    elif pair == {"G", "K"}:
        E = 9.0 * K * G / (3.0 * K + G)
        nu = E / (2.0 * G) - 1.0
    elif pair == {"G", "nu"}:
        E = 2.0 * G * (1.0 + nu)
        K = E / (3.0 * (1.0 - 2.0 * nu))
    elif pair == {"K", "nu"}:
        E = 3.0 * K * (1.0 - 2.0 * nu)
        G = E / (2.0 * (1.0 + nu))

    return {"E": E, "nu": nu, "G": G, "K": K}


def is_spd(matrix):
    """Check if a matrix is Symmetric Positive Definite"""
    # Check symmetry
    is_symmetric = np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8)

    # Check positive definiteness
    eigenvalues = eigvalsh(matrix)
    is_positive_definite = np.all(eigenvalues > 0)

    return is_symmetric and is_positive_definite, eigenvalues


def compute_VoigtReuss_bounds(phase_tensors, volume_fractions):
    """
    Compute Voigt and Reuss bounds from phase tensors and volume fractions.

    Parameters
    ----------
    phase_tensors : list
        List of phase-wise tensors
    volume_fractions : list
        List of volume fractions for each phase

    Returns
    -------
    voigt : ndarray
        Voigt bound tensor
    reuss : ndarray
        Reuss bound tensor
    """
    # Convert inputs to numpy arrays for vectorized operations
    phase_tensors = np.array(phase_tensors)
    volume_fractions = np.array(volume_fractions)

    # Voigt bound (arithmetic mean)
    voigt = np.sum(volume_fractions[:, np.newaxis, np.newaxis] * phase_tensors, axis=0)

    # Reuss bound (harmonic mean)
    phase_inverses = np.array([np.linalg.inv(tensor) for tensor in phase_tensors])
    reuss_inv = np.sum(
        volume_fractions[:, np.newaxis, np.newaxis] * phase_inverses, axis=0
    )
    reuss = np.linalg.inv(reuss_inv)

    return voigt, reuss
