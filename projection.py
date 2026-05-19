"""
Projection module - 3D to 2D screen projection functions.
No pygame dependencies.
"""

from config import CONFIG, SCREEN_W, SCREEN_H


def project_3d_to_screen(x, y, z, width=None, height=None):
    """Project 3D coordinates to 2D screen (perspective projection).
    
    Uses perspective divide: objects farther from the camera appear smaller.
    Returns None if the point is behind the camera.
    
    Args:
        x, y, z: 3D coordinates
        width: screen width (defaults to SCREEN_W from config)
        height: screen height (defaults to SCREEN_H from config)
    """
    if width is None:
        width = SCREEN_W
    if height is None:
        height = SCREEN_H
    
    camera_distance = CONFIG['projection'].get('camera_distance', 15.0)
    fov = CONFIG['projection'].get('fov', 500.0)
    
    # Clip points behind the camera
    z_depth = z + camera_distance
    if z_depth < 0.1:  # Too close to or behind the camera
        return None
    
    # Perspective divide
    scale = fov / z_depth
    px = int(width // 2 + x * scale)
    py = int(height // 2 + y * scale)
    
    return (px, py)
