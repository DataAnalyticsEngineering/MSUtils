import math
import meshio
import numpy as np
import trimesh
from scipy.special import gamma
from scipy.spatial import ConvexHull
from pyrecest.sampling.hyperspherical_sampler import LeopardiSampler as LeopardiSampler

if __name__ == '__main__':

    num_points = 500
    dim = 3
    
    sampler = LeopardiSampler(original_code_column_order=True)
    grid, description = sampler.get_grid(num_points, dim-1)
    print("\nLeopardi sampler grid:")
    print(grid.round(4))


    
    if dim == 3:
        # Create a sphere of radius 1 centered at (0,0,0).
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
        # Convert the trimesh sphere to a meshio Mesh.
        sphere_mesh = meshio.Mesh(points=sphere.vertices, cells=[("triangle", sphere.faces)])
        meshio.write("data/unit_sphere.vtk", sphere_mesh)
        
        # Create a cell block of type "vertex" where each grid point becomes a vertex.
        cells = [("vertex", np.arange(grid.shape[0]).reshape(-1, 1))]

        # Create the mesh for the Leopardi grid and write it to a VTK file.
        mesh = meshio.Mesh(points=grid, cells=cells)
        meshio.write("data/leopardi_grid.vtk", mesh)

        # Use the convex hull of the grid points to get a triangulation of the sphere surface.
        hull = ConvexHull(grid)
        triangles = hull.simplices  # each row contains indices for one triangle
        
        triangle_areas = []
        for tri in triangles:
            # Retrieve vertices and ensure they lie on the unit sphere.
            A, B, C = grid[tri]
            A = A / np.linalg.norm(A)
            B = B / np.linalg.norm(B)
            C = C / np.linalg.norm(C)

            # Compute the arc lengths (central angles) between pairs of vertices.
            a = np.arccos(np.clip(np.dot(B, C), -1.0, 1.0))
            b = np.arccos(np.clip(np.dot(A, C), -1.0, 1.0))
            c = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))

            # Compute the internal angles of the spherical triangle using the spherical law of cosines.
            angle_A = np.arccos(np.clip((np.cos(a) - np.cos(b)*np.cos(c)) / (np.sin(b)*np.sin(c)), -1.0, 1.0))
            angle_B = np.arccos(np.clip((np.cos(b) - np.cos(a)*np.cos(c)) / (np.sin(a)*np.sin(c)), -1.0, 1.0))
            angle_C = np.arccos(np.clip((np.cos(c) - np.cos(a)*np.cos(b)) / (np.sin(a)*np.sin(b)), -1.0, 1.0))

            # Girard's theorem: Area = spherical excess for a unit sphere.
            spherical_excess = angle_A + angle_B + angle_C - np.pi
            triangle_areas.append(spherical_excess)
        
        triangle_areas = np.array(triangle_areas)
        total_triangle_area = triangle_areas.sum()
        
        # Surface area of a unit hypersphere of dimension n is given by: A = 2 * pi^(n/2) / Gamma(n/2)
        # https://mathworld.wolfram.com/Hypersphere.html
        unit_sphere_surface_area = 2 * math.pi**((dim)/2) / gamma((dim)/2)

        print("\nTotal area from convex hull triangulation:", total_triangle_area)
        print(f"Surface area of a unit sphere in {dim+1} dimensions:", unit_sphere_surface_area)
        print(f"Difference: {np.abs(total_triangle_area - unit_sphere_surface_area):.10f}")
        
  
  
  
  