import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, Delaunay


def is_inside_polyhedron(points, polyhedron_points):
    deln = Delaunay(polyhedron_points)
    return deln.find_simplex(points) >= 0


# Generate more complex polyhedron vertices
polyhedron_points = np.random.rand(30, 3) * 2 - 1  # 30 random points in 3D space
hull = ConvexHull(polyhedron_points)


# Generate and check points, and measure execution time
def generate_and_check_points(num_points):
    # Generate random points within a bounding box around the polyhedron
    bounding_box_min = np.min(polyhedron_points, axis=0) - 0.5
    bounding_box_max = np.max(polyhedron_points, axis=0) + 0.5
    random_points = np.random.uniform(
        low=bounding_box_min, high=bounding_box_max, size=(num_points, 3)
    )

    # Measure time to check points
    start_time = time.time()
    inside = is_inside_polyhedron(random_points, polyhedron_points)
    end_time = time.time()

    return random_points, inside, end_time - start_time


# Test with increasing number of points
point_counts = [
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
]  # Adjust based on your testing needs
for count in point_counts:
    random_points, inside, duration = generate_and_check_points(count)
    print(f"{count} points: {duration:.4f} seconds, {np.sum(inside)} points inside")

# Visualization for the last test case
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot polyhedron vertices and faces
ax.scatter(
    polyhedron_points[:, 0],
    polyhedron_points[:, 1],
    polyhedron_points[:, 2],
    color="g",
    s=50,
    label="Polyhedron Vertices",
)
faces = [polyhedron_points[simplex] for simplex in hull.simplices]
face_collection = Poly3DCollection(
    faces, alpha=0.25, linewidths=1, edgecolors="k", facecolors="cyan"
)
ax.add_collection3d(face_collection)

# Plot all random points
ax.scatter(
    random_points[:, 0],
    random_points[:, 1],
    random_points[:, 2],
    color="b",
    s=1,
    alpha=0.5,
    label="Random Points",
)

# Highlight points inside the polyhedron
points_inside = random_points[inside]
ax.scatter(
    points_inside[:, 0],
    points_inside[:, 1],
    points_inside[:, 2],
    color="r",
    s=10,
    label="Points Inside",
)

ax.legend()
plt.show()


# from scipy.spatial import ConvexHull
# import numpy as np
# import itertools

# def _is_inside_polyhedron(point, polyhedron_points):
#         hull = ConvexHull(polyhedron_points)
#         new_hull = ConvexHull(np.vstack((polyhedron_points, point)))
#         return np.array_equal(new_hull.vertices, hull.vertices)  # Point is inside if hull doesn't change

# if __name__ == "__main__":
#     point = np.array([0, 0, 0])
#     # Generate all coordinates of a unit cube
#     polyhedron_points = list(itertools.product([0, 1], repeat=3))

#     # Print the coordinates
#     for coordinate in polyhedron_points:
#         print(coordinate)
#     print(_is_inside_polyhedron(point, polyhedron_points))

#     point = np.array([0.5, 0.5, 0.75])
#     print(_is_inside_polyhedron(point, polyhedron_points))
#     point = np.array([0.5, 0.5, 1])
#     print(_is_inside_polyhedron(point, polyhedron_points))
#     point = np.array([0.5, 0.5, -1])
#     print(_is_inside_polyhedron(point, polyhedron_points))
#     point = np.array([0.5, 0.5, 0])
#     print(_is_inside_polyhedron(point, polyhedron_points))
