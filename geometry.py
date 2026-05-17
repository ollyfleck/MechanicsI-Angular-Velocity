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

# All 6 cube face definitions with their normals (for 3D rendering)
# Vertex layout from CUBE_VERTS:
#   Vertex 0: (+size, +size, +size)  - top-right-front
#   Vertex 1: (-size, +size, +size)  - top-left-front
#   Vertex 2: (-size, -size, +size)  - bottom-left-front
#   Vertex 3: (+size, -size, +size)  - bottom-right-front
#   Vertex 4: (+size, +size, -size)  - top-right-back
#   Vertex 5: (-size, +size, -size)  - top-left-back
#   Vertex 6: (-size, -size, -size)  - bottom-left-back
#   Vertex 7: (+size, -size, -size)  - bottom-right-back
CUBE_FACES_3D = [
    {'indices': (0, 1, 2, 3), 'name': 'front',   'normal': (0, 0, 1),   'color': (100, 255, 100)},   # Green - +Z face
    {'indices': (4, 7, 6, 5), 'name': 'back',    'normal': (0, 0, -1),  'color': (100, 200, 255)},   # Cyan - -Z face
    {'indices': (0, 3, 7, 4), 'name': 'right',   'normal': (1, 0, 0),   'color': (255, 150, 100)},   # Orange - +X face
    {'indices': (1, 5, 6, 2), 'name': 'left',    'normal': (-1, 0, 0),  'color': (200, 100, 255)},   # Purple - -X face
    {'indices': (0, 4, 5, 1), 'name': 'top',     'normal': (0, 1, 0),   'color': (255, 255, 100)},   # Yellow - +Y face
    {'indices': (3, 2, 6, 7), 'name': 'bottom',  'normal': (0, -1, 0),  'color': (255, 100, 200)},   # Pink - -Y face
]


def compute_face_center(vertex_indices):
    """Compute the center point of a face given its vertex indices."""
    int_indices = [int(i) for i in vertex_indices]
    face_verts_list = [CUBE_VERTS[i] for i in int_indices]
    return np.mean(face_verts_list, axis=0)