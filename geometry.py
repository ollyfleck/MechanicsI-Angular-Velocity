"""
Geometry module - Cube geometry definitions (vertices, edges, faces).
No pygame or physics dependencies.
"""

import numpy as np

from config import CUBE_SIZE


# ==================== CUBE GEOMETRY ====================

# Cube vertices (8 corners)
CUBE_VERTS = np.array([
    [ CUBE_SIZE,   CUBE_SIZE,   CUBE_SIZE],  # 0: top-right-front
    [-CUBE_SIZE,   CUBE_SIZE,   CUBE_SIZE],  # 1: top-left-front  
    [-CUBE_SIZE, -CUBE_SIZE,   CUBE_SIZE],  # 2: top-left-back
    [ CUBE_SIZE, -CUBE_SIZE,   CUBE_SIZE],  # 3: top-right-back
    [ CUBE_SIZE,   CUBE_SIZE,  -CUBE_SIZE],  # 4: bottom-right-front
    [-CUBE_SIZE,   CUBE_SIZE,  -CUBE_SIZE],  # 5: bottom-left-front  
    [-CUBE_SIZE, -CUBE_SIZE,  -CUBE_SIZE],  # 6: bottom-left-back
    [ CUBE_SIZE, -CUBE_SIZE,  -CUBE_SIZE],  # 7: bottom-right-back
])

# Cube edges (indices connecting vertices)
CUBE_EDGES = [
    (0,1),(1,2),(2,3),(3,0),  # Top face
    (4,5),(5,6),(6,7),(7,4),  # Bottom face
    (0,4),(1,5),(2,6),(3,7)   # Connecting edges
]

# Face indices for velocity vectors
CUBE_FACE_INDICES = [
    (0, 1, 5, 4),  # Front (z = +1)
    (2, 3, 7, 6),  # Back (z = -1)  
    (0, 3, 7, 4),  # Right (x = +1)
    (1, 2, 6, 5),  # Left (x = -1)
]


def compute_face_center(vertex_indices):
    """Compute the center point of a face given its vertex indices."""
    int_indices = [int(i) for i in vertex_indices]
    face_verts_list = [CUBE_VERTS[i] for i in int_indices]
    return np.mean(face_verts_list, axis=0)