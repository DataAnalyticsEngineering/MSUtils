import numpy as np
import time

def euclidean_dist_matrix(A, B):
    """Compute the squared Euclidean distance matrix using the formula."""
    l_A = np.linalg.norm(A, axis=1)
    l_B = np.linalg.norm(B, axis=1)
    d_AB = (l_A**2)[:, None] + (l_B**2)[None, :] - 2.*A@B.T
    return d_AB

def periodic_dist_matrix(A, B, RVE_length):
    """Compute periodic distance matrix between points in A and B."""
    # Difference between each combination of points
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    
    # Handle periodicity
    diff %= RVE_length
    diff = np.where(diff > 0.5 * RVE_length, diff - RVE_length, diff)
    
    # Squared distances
    dist_sq = np.sum(diff**2, axis=-1)
    return dist_sq


A = np.random.uniform(0, 1, size=(100000, 3))
B = np.random.uniform(0, 1, size=(1000, 3))

# Time the Euclidean distance approach
start_time = time.time()
d_AB = euclidean_dist_matrix(A, B)
euclidean_time = time.time() - start_time
print(f"Euclidean distance computation took {euclidean_time} seconds.")
idx_euclidean = np.argmin(d_AB, axis=1)

# Time the periodic distance approach
start_time = time.time()
d_periodic = periodic_dist_matrix(A, B, np.array([100,100,100]))
periodic_time = time.time() - start_time
print(f"Periodic distance computation took {periodic_time} seconds.")
idx_periodic = np.argmin(d_periodic, axis=1)

# Check if indices are the same
are_idx_same = np.array_equal(idx_euclidean, idx_periodic)
print(f"idx_euclidean and idx_periodic are the same: {are_idx_same}")

# Check if distance matrices are identical
are_dists_same = np.array_equal(d_AB, d_periodic)
print(f"d_AB and d_periodic are identical: {are_dists_same}")

# Check if distance matrices are statistically similar
mean_diff = np.abs(np.mean(d_AB) - np.mean(d_periodic))
std_diff = np.abs(np.std(d_AB) - np.std(d_periodic))
print(f"Mean difference between d_AB and d_periodic: {mean_diff}")
print(f"Standard deviation difference between d_AB and d_periodic: {std_diff}")

# Check for any significant differences in the matrices
significant_diff_count = np.sum(np.abs(d_AB - d_periodic) > 1e-6) 
print(f"Number of significantly different entries: {significant_diff_count}")

# For further verification, display differences in the matrices
if not are_dists_same:
    differences = d_AB - d_periodic
    print("\nDifferences between d_AB and d_periodic:")
    print(differences)

    # Display the indices where the two matrices differ significantly
    significant_diff_indices = np.argwhere(np.abs(differences) > 1e-6)
    print("\nIndices of significantly different entries:")
    print(significant_diff_indices)

