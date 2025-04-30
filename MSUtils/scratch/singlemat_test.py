import numpy as np
import h5py

# Create the 3D image matrix of ints with all elements set to -1
image_matrix = np.ones((4, 4, 4), dtype=int) * -1

# Create 3 matrices representing vectors (GB normals)
GBNormals_X = np.ones((4, 4, 4, 3), dtype=float) * np.array([1, 0, 0])
GBNormals_Y = np.ones((4, 4, 4, 3), dtype=float) * np.array([0, 1, 0])
GBNormals_Z = np.ones((4, 4, 4, 3), dtype=float) * np.array([0, 0, 1])

# Save all matrices to an h5 file
with h5py.File('data/single_material_image.h5', 'w') as f:
    f.create_dataset('image', data=image_matrix)
    f.create_dataset('GBNormals_X', data=GBNormals_X)
    f.create_dataset('GBNormals_Y', data=GBNormals_Y)
    f.create_dataset('GBNormals_Z', data=GBNormals_Z)