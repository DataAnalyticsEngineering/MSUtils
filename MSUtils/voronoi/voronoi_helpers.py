import numpy as np
from sympy.ntheory import factorint

def periodic_difference(pt1, pt2, RVE_length):
    """Compute the periodic difference vector and its norm between two points."""
    diff = pt2 - pt1
    diff = np.where(diff > 0.5 * RVE_length, diff - RVE_length, diff)
    diff = np.where(diff < -0.5 * RVE_length, diff + RVE_length, diff)
    norm_diff = np.linalg.norm(diff)
    return diff, norm_diff

def periodic_dist_matrix(A, B, RVE_length):
        """Compute periodic distance matrix between points in A and B."""
        # Expand dimensions
        A = A[:, np.newaxis, :]
        B = B[np.newaxis, :, :]

        # Compute the periodic difference between all combinations of points in A and B
        diff = A - B
        diff = np.where(diff > 0.5 * RVE_length, diff - RVE_length, diff)
        diff = np.where(diff < -0.5 * RVE_length, diff + RVE_length, diff)

        # Compute the squared distances
        dist_sq = np.sum(diff**2, axis=-1)
        return dist_sq

from scipy.ndimage import binary_erosion
def periodic_erosion(mask, shrink_factor):
    """Applies erosion on a binary mask while considering periodic boundaries."""
    selem = np.ones((shrink_factor, shrink_factor, shrink_factor))
    padded_mask = np.pad(mask, ((shrink_factor//2, shrink_factor//2), 
                                (shrink_factor//2, shrink_factor//2), 
                                (shrink_factor//2, shrink_factor//2)), mode='wrap')
    eroded_padded_mask = binary_erosion(padded_mask, structure=selem)
    Nx, Ny, Nz = mask.shape
    return eroded_padded_mask[shrink_factor//2:Nx+shrink_factor//2, 
                              shrink_factor//2:Ny+shrink_factor//2, 
                              shrink_factor//2:Nz+shrink_factor//2]

def calculate_polygon_area_3d(vertices, normal):
    """
    Calculate the area of a polygon in 3D.

    Parameters:
    - vertices: List of 3D coordinates representing the vertices of the polygon.
    - normal: 3D vector representing the normal of the polygon.

    Returns:
    - area: The area of the polygon.

    This function calculates the area of a polygon in 3D by projecting the vertices onto a plane spanned by two orthogonal vectors and then using the Shoelace formula to calculate the area in 2D.
    """
    # Ensure normal is a unit vector
    normal = normal / np.linalg.norm(normal)

    # Find a vector that is not parallel to the normal
    if not np.isclose(normal[0], 0) or not np.isclose(normal[1], 0):
        non_parallel_vector = np.array([0, 0, 1])
    else:
        non_parallel_vector = np.array([1, 0, 0])

    # Find two vectors that are orthogonal to the normal
    u = np.cross(normal, non_parallel_vector)
    v = np.cross(normal, u)

    # Normalize u and v to form an orthonormal basis with the normal
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # Project vertices onto the plane spanned by u and v
    vertices_2d = np.array([[np.dot(vertex - vertices[0], u), np.dot(vertex - vertices[0], v)] for vertex in vertices])

    # Use the Shoelace formula to calculate the area of the polygon in 2D
    x, y = vertices_2d[:, 0], vertices_2d[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def factorize(n, dim):
    """
    Use sympy's factorint to factorize `n` into prime factors and then group them into `dim` factors
    that are as close as possible.

    :param n: The number to factorize.
    :param dim: The number of factors to group into (e.g., 3 for 3D).
    :return: A list of `dim` factors whose product equals `n`.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1.")
    
    # Get prime factorization of n
    prime_factors = factorint(n)
    
    # Start with `dim` equal factors
    factors = [1] * dim
    
    # Sort the factors in descending order to distribute larger primes first
    primes = sorted((prime for prime, exp in prime_factors.items() for _ in range(exp)), reverse=True)
    
    # Distribute the primes among the factors to minimize the difference between factors
    for prime in primes:
        # Find the factor with the smallest value and multiply it by the current prime
        smallest_idx = factors.index(min(factors))
        factors[smallest_idx] *= prime
    
    factors.sort(reverse=True)
    return factors
