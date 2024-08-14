import numpy as np
from voronoi_helpers import calculate_polygon_area_3d

from scipy.spatial import Voronoi, ConvexHull, Delaunay
class PeriodicVoronoiTessellation:
    def __init__(self, RVE_size, seeds):
        self.RVE_size = RVE_size
        self.ndim = len(RVE_size)
        self.seeds = seeds
        self._generate_voronoi()
        self._characterize_voronoi()
        self.neighbors, _ = self._find_neighbors()

        # Renumber vertices for continuity
        unique_vertex_indices = sorted(
            set(index for region in self.orig_regions for index in region if index != -1)
        )
        self.vertex_index_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)
        }

    def _extend_seeds(self):
        extended, orig_indices = [], []
        dim = len(self.RVE_size)  # Determine the dimension (2D or 3D)

        # Initialize a dictionary to map from every extended seed index to the original seed index
        crystal_index_map = {}

        for seed_idx, seed in enumerate(self.seeds):
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    z_range = [-1, 0, 1] if dim == 3 else [0]  # Include z-loop for 3D
                    for z in z_range:
                        displacement = np.array(
                            [x * self.RVE_size[0], y * self.RVE_size[1]] +
                            ([z * self.RVE_size[2]] if dim == 3 else [])
                        )
                        extended_seed = seed + displacement
                        extended.append(extended_seed)

                        extended_seed_idx = len(extended) - 1  # Index of the newly added extended seed
                        # Map every extended seed index to the original seed index
                        crystal_index_map[extended_seed_idx] = seed_idx

                        if (x, y, z)[:dim] == tuple([0] * dim):  # Original seed position in n-dimensions
                            orig_indices.append(extended_seed_idx)

        self.crystal_index_map = crystal_index_map
        return np.array(extended), np.array(orig_indices)
    
    def _generate_voronoi(self):
        ext_seeds, orig_indices = self._extend_seeds()
        self.voronoi = Voronoi(ext_seeds)

        self.orig_indices = orig_indices
        self.orig_regions = [
            self.voronoi.regions[self.voronoi.point_region[i]] for i in orig_indices
        ]
        self.orig_verts = np.array(
            list(set().union(*(r for r in self.orig_regions if -1 not in r)))
        )

        orig_ridge_idx = [
            i
            for i, v in enumerate(self.voronoi.ridge_vertices)
            if all(v in self.orig_verts for v in v)
        ]
        self.orig_ridges = [self.voronoi.ridge_vertices[i] for i in orig_ridge_idx]
        self.orig_ridge_pts = np.array(
            [self.voronoi.ridge_points[i] for i in orig_ridge_idx]
        )
        self.orig_ridge_indices = np.array(orig_ridge_idx)

    def _find_neighbors(self):
        # Create replicated points across the domain for periodic boundary conditions
        offsets = np.array([
            [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
            [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
            [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],

            [0, -1, -1], [0, -1, 0], [0, -1, 1],
            [0, 0, -1],                [0, 0, 1],
            [0, 1, -1], [0, 1, 0], [0, 1, 1],

            [1, -1, -1], [1, -1, 0], [1, -1, 1],
            [1, 0, -1], [1, 0, 0], [1, 0, 1],
            [1, 1, -1], [1, 1, 0], [1, 1, 1],
        ])
        seeds = self.seeds
        RVE_length = self.RVE_size
        replicated_seeds = [seeds + offset * RVE_length for offset in offsets]
        all_seeds = np.vstack([seeds] + replicated_seeds)
        
        # Apply Delaunay triangulation
        tri = Delaunay(all_seeds)
        
        # Extract neighboring relationships
        neighbors = {}
        for simplex in tri.simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        if i < len(seeds):  # Ensure we're looking only at original seeds, not replicas
                            if i not in neighbors:
                                neighbors[i] = set()
                            # Map replicas back to their original counterparts
                            # neighbors[i].add(j % len(seeds))
                            mapped_j = j % len(seeds)
                            if i != mapped_j:
                                neighbors[i].add(mapped_j)
        
        # Convert sets to lists
        for key in neighbors:
            neighbors[key] = sorted(list(neighbors[key]))

        return neighbors, tri

    def _characterize_voronoi(self):
        total_area = 0        
        Ltensor = np.zeros((self.ndim, self.ndim))
        LLtensor = np.zeros((self.ndim, self.ndim, self.ndim, self.ndim))

        for ridge, ridge_pts in zip(self.orig_ridges, self.orig_ridge_pts):
            if -1 in ridge:  # Skip ridges at infinity
                continue
            
            # if any(pt not in self.orig_indices for pt in ridge_pts): # Skip boundary ridges
            #     continue
            
            vertices = [self.voronoi.vertices[i] for i in ridge]
            ridge_N = self.voronoi.points[ridge_pts[0]] - self.voronoi.points[ridge_pts[1]]
            ridge_N /= np.linalg.norm(ridge_N)

            if self.ndim == 2:  # 2D case
                ridge_area = np.linalg.norm(vertices[1] - vertices[0])
                
            elif self.ndim == 3:  # 3D case
                ridge_area = calculate_polygon_area_3d(vertices, ridge_N)
            
            total_area += ridge_area
            Ltensor += ridge_area * np.einsum('i,j->ij', ridge_N, ridge_N)
            LLtensor += ridge_area * np.einsum('i,j,k,l->ijkl', ridge_N, ridge_N, ridge_N, ridge_N)
        
        crystal_volumes = np.zeros(len(self.orig_regions))
        for i, region in enumerate(self.orig_regions):
            crystal_volumes[i] = ConvexHull([self.voronoi.vertices[j] for j in region]).volume

        self.crystal_volumes = crystal_volumes
        self.interface_area = total_area
        self.Ltensor = Ltensor
        self.LLtensor = LLtensor

    def write_to_vtu(self, filename):
        vertices = self.voronoi.vertices
        crystals = self.orig_indices
        regions = self.orig_regions
        ridges = self.orig_ridges
        ridge_points = self.orig_ridge_pts
        dim = self.seeds.shape[1]

        # Renumber vertices for continuity
        unique_vertex_indices = sorted(
            set(index for region in regions for index in region if index != -1)
        )
        vertex_index_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)
        }

        crystal_index_map = {
            crystal_idx: new_idx for new_idx, crystal_idx in enumerate(crystals)
        }
        with open(filename, "w") as file:
            # Write VTU header
            file.write(
                '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
            )
            file.write("<UnstructuredGrid>\n")
            file.write(
                f'<Piece NumberOfPoints="{len(unique_vertex_indices)}" NumberOfCells="{len(regions)}">\n'
            )

            # Write points (vertices)
            file.write("<Points>\n")
            file.write('<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            for idx in unique_vertex_indices:
                v = vertices[idx]
                file.write(f"{v[0]} {v[1]} {v[2] if dim == 3 else 0}\n")
            file.write("</DataArray>\n")
            file.write("</Points>\n")

            # Write cells
            file.write("<Cells>\n")
            # Connectivity
            file.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for region in regions:
                renumbered_region = [vertex_index_map[v] for v in region if v != -1]
                file.write(" ".join(str(v) for v in renumbered_region) + "\n")
            file.write("</DataArray>\n")
            # Offsets
            file.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
            offset = 0
            for region in regions:
                offset += len([v for v in region if v != -1])
                file.write(f"{offset}\n")
            file.write("</DataArray>\n")
            # Types
            file.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
            for _ in regions:
                file.write(
                    "42\n" if dim == 3 else "7\n"
                )  # 42 for polyhedron, 7 for polygon
            file.write("</DataArray>\n")

            # For 3D, write faces and face offsets
            if dim == 3:
                lofe_ids = [[] for _ in range(len(regions))]

                # Iterate through each ridge and the corresponding ridge points
                for ridge, pts in zip(ridges, ridge_points):
                    # Add the ridge to the corresponding cells' face list
                    for pt in pts:
                        if pt in crystals:
                            lofe_ids[crystal_index_map[pt]].append(ridge)

                file.write('<DataArray type="Int32" Name="faces" format="ascii">\n')
                for cell_faces in lofe_ids:
                    n_faces = len(cell_faces)
                    file.write(f"{n_faces} ")  # Number of faces for this cell
                    for face in cell_faces:
                        n_vertices = len(face)
                        file.write(f"{n_vertices} ")  # Number of vertices for this face
                        renumbered_face = [vertex_index_map[v] for v in face]
                        file.write(" ".join(str(v) for v in renumbered_face) + " ")
                    file.write("\n")
                file.write("</DataArray>\n")

                # Write face offsets
                file.write('<DataArray type="Int32" Name="faceoffsets" format="ascii">\n')
                cumulative_offset = 0
                for cell_faces in lofe_ids:
                    for face in cell_faces:
                        cumulative_offset += len(face) + 1  # +1 for the face size entry
                    cumulative_offset += 1  # +1 for the number of faces entry
                    file.write(f"{cumulative_offset}\n")
                file.write("</DataArray>\n")

            file.write("</Cells>\n")

            # Write crystal IDs as cell data
            file.write('<CellData Scalars="crystal_id">\n')
            file.write('<DataArray type="Int32" Name="crystal_id" format="ascii">\n')
            for crystal_id in range(len(regions)):
                file.write(f"{crystal_id}\n")
            file.write("</DataArray>\n")
            file.write("</CellData>\n")

            file.write("</Piece>\n")
            file.write("</UnstructuredGrid>\n")
            file.write("</VTKFile>\n")