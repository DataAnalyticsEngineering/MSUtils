import numpy as np
import plotly.graph_objects as go


def BCC_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    edges = [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)]
    return vertices, edges


def BCCz_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 8),
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return vertices, edges


def cubic_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return vertices, edges


def FCC_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1],
        ],
        dtype=np.float64,
    )

    edges = [
        (0, 8),
        (1, 8),
        (3, 8),
        (2, 8),
        (2, 12),
        (3, 12),
        (6, 12),
        (7, 12),
        (0, 10),
        (4, 10),
        (3, 10),
        (7, 10),
        (0, 9),
        (1, 9),
        (4, 9),
        (5, 9),
        (4, 13),
        (7, 13),
        (6, 13),
        (5, 13),
        (1, 11),
        (5, 11),
        (6, 11),
        (2, 11),
    ]

    return vertices, edges


def FBCC_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    edges = [
        (0, 8),
        (1, 8),
        (3, 8),
        (2, 8),
        (2, 12),
        (3, 12),
        (6, 12),
        (7, 12),
        (0, 10),
        (4, 10),
        (3, 10),
        (7, 10),
        (0, 9),
        (1, 9),
        (4, 9),
        (5, 9),
        (4, 13),
        (7, 13),
        (6, 13),
        (5, 13),
        (1, 11),
        (5, 11),
        (6, 11),
        (2, 11),
        (0, 14),
        (1, 14),
        (2, 14),
        (3, 14),
        (4, 14),
        (5, 14),
        (6, 14),
        (7, 14),
    ]

    return vertices, edges


def isotruss_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 1, 0.5],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (0, 8),
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 9),
        (8, 10),
        (8, 11),
        (8, 12),
        (8, 13),
        (8, 14),
    ]
    return vertices, edges


def octet_truss_lattice():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # Bottom square
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],  # Top square
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],  # Midpoints of top and bottom faces
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],  # Midpoints of vertical edges
            [0.5, 0, 0.5],
            [0.5, 1, 0.5],  # Midpoints of horizontal edges
        ],
        dtype=np.float64,
    )

    edges = [
        (0, 8),
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 9),
        (5, 9),
        (6, 9),
        (7, 9),
        (3, 10),
        (0, 10),
        (7, 10),
        (4, 10),
        (2, 11),
        (1, 11),
        (6, 11),
        (5, 11),
        (0, 12),
        (1, 12),
        (4, 12),
        (5, 12),
        (2, 13),
        (3, 13),
        (6, 13),
        (7, 13),
        (11, 12),
        (12, 10),
        (10, 13),
        (13, 11),
        (8, 11),
        (11, 9),
        (9, 10),
        (10, 8),
        (8, 13),
        (13, 9),
        (9, 12),
        (12, 8),
    ]
    return vertices, edges


def plot_lattice(vertices, edges):
    # Create a scatter plot for vertices
    scatter_vertices = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers+text",
        marker=dict(size=5, color="red"),
        text=[str(i) for i in range(len(vertices))],
        textposition="top center",
        textfont=dict(size=20, color="black"),
    )

    # Create a list for edge lines and edge numbers
    edge_lines = []
    edge_numbers = []
    for j, edge in enumerate(edges):
        start, end = edge
        line = np.array([vertices[start], vertices[end]])
        edge_lines.append(
            go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )
        mid_point = (vertices[start] + vertices[end]) / 2
        edge_numbers.append(
            go.Scatter3d(
                x=[mid_point[0]],
                y=[mid_point[1]],
                z=[mid_point[2]],
                mode="markers+text",
                text=[str(j)],
                textposition="middle center",
                marker=dict(size=1, color="green"),
                textfont=dict(size=20, color="green"),
                showlegend=False,
            )
        )

    # Combine scatter plot and edge lines
    fig = go.Figure(data=[scatter_vertices] + edge_lines + edge_numbers)

    # Set axis labels
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    fig.show()


def check_rigidity(vertices, edges):
    n = len(vertices)  # number of nodes
    m = len(edges)  # number of struts
    result = m - 3 * n + 6
    if result == 0:
        return "Lattice is statically determinate and rigid"
    elif result > 0:
        return "Lattice is statically indeterminate"
    else:
        return "Lattice is flexible and unstable"
