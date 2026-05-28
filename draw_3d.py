"""
3D Drawing module - Enhanced 3D drawing functions with depth sorting and shading.
Requires pygame.
"""

import math
import numpy as np

try:
    import pygame
except ImportError:
    raise RuntimeError("pygame is required for drawing functions. Run: pip install pygame")

from config import COLORS, CONFIG, CUBE_SIZE, SCREEN_W, SCREEN_H
from projection import project_3d_to_screen
from math_utils import cross_product, rotate_around_axis


# ==================== VECTOR SHADING ====================

def compute_vector_alpha(magnitude, tip_length):
    """Compute alpha (0-255) for a vector based on its magnitude relative to tip length.
    
    Fade curve:
      - magnitude 0        -> alpha 0   (fully transparent)
      - magnitude < tip    -> alpha = (magnitude / tip) * 255  (linear fade)
      - magnitude >= tip   -> alpha 255 (fully opaque, shaft exists)
    
    Args:
        magnitude: current display length of the vector
        tip_length: the tip/arrowhead length from config
    
    Returns:
        Alpha value 0-255, or None if fully opaque (no alpha blending needed).
        Returns None when alpha blending is disabled globally.
    """
    # Respect global alpha blending toggle
    vr = CONFIG.get('vector_rendering', {})
    if not vr.get('alpha_blending_enabled', True):
        return None
    
    if tip_length <= 0:
        return None
    if magnitude >= tip_length:
        return None  # Fully opaque, no alpha blending needed
    alpha = (magnitude / tip_length) * 255.0
    return max(0, min(255, int(alpha)))


def apply_vector_shading_to_face(color, face_normal, depth):
    """Apply simple Phong-like shading to a face color based on its normal and depth.
    
    Returns shaded (r, g, b) tuple.
    """
    shading_config = CONFIG.get('vector_shading', {})
    if not shading_config.get('enabled', True):
        return color
    
    ambient = shading_config.get('ambient', 0.35)
    diffuse_max = shading_config.get('diffuse_max', 0.65)
    light_dir = shading_config.get('light_direction', [0.5, 0.8, 0.3])
    
    # Normalize light direction
    light_mag = math.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
    if light_mag < 1e-10:
        return color
    lx, ly, lz = light_dir[0]/light_mag, light_dir[1]/light_mag, light_dir[2]/light_mag
    
    # Normalize face normal
    nx, ny, nz = face_normal[0], face_normal[1], face_normal[2]
    n_mag = math.sqrt(nx*nx + ny*ny + nz*nz)
    if n_mag < 1e-10:
        return color
    nx, ny, nz = nx/n_mag, ny/n_mag, nz/n_mag
    
    # Dot product (light facing toward camera = positive Z in our coord system)
    # Use the face normal's Z component to determine if it's facing the viewer
    facing_factor = max(0, nz)  # Simple front-face detection
    
    # Diffuse lighting based on angle to light
    diffuse = facing_factor * diffuse_max
    
    # Depth-based ambient boost (further objects get slightly brighter)
    depth_factor = min(max(depth / 20.0, 0.0), 1.0)
    effective_ambient = ambient * (0.7 + 0.3 * depth_factor)
    
    r = min(255, int(color[0] * (effective_ambient + diffuse)))
    g = min(255, int(color[1] * (effective_ambient + diffuse)))
    b = min(255, int(color[2] * (effective_ambient + diffuse)))
    
    return (r, g, b)


# ==================== PERFORMANCE UTILITIES ====================

def compute_dynamic_segments(screen_segments, start_3d, end_3d, fov_factor=500.0, camera_distance=25.0, screen_width=1000, screen_height=1000):
    """Compute an appropriate segment count based on the vector's screen-space size.
    
    For short vectors that project to a small screen size, use fewer segments
    to improve performance without visible quality loss.
    
    Returns: (segment_count, projected_length)
    """
    # Project start and end to screen (use actual screen dimensions, not scale factors)
    start_screen = project_3d_to_screen(start_3d[0], start_3d[1], start_3d[2], screen_width, screen_height)
    end_screen = project_3d_to_screen(end_3d[0], end_3d[1], end_3d[2], screen_width, screen_height)
    
    if start_screen is None or end_screen is None:
        return (screen_segments, 0.0)
    
    screen_dx = end_screen[0] - start_screen[0]
    screen_dy = end_screen[1] - start_screen[1]
    screen_length = math.sqrt(screen_dx * screen_dx + screen_dy * screen_dy)
    
    # Scale segments based on screen length:
    # - Less than 10px: 4 segments (minimum for roundness)
    # - 10-30px: linearly interpolate from 4 to screen_segments
    # - More than 30px: full segment count
    if screen_length < 10:
        return (4, screen_length)
    if screen_length > 30:
        return (screen_segments, screen_length)
    
    # Linear interpolation
    return (max(4, int(screen_segments * (screen_length - 10) / 20)), screen_length)


# ==================== 3D PRIMITIVE DRAWING ====================

def draw_3d_cylinder(start, end, radius, segments, screen, color, depth_offset=0.0, width=None, height=None):
    """Draw a 3D cylinder as filled polygons, depth-sorted.
    
    Draws a cylinder from start to end with given radius and segment count.
    Returns list of faces as ((p1, p2, p3, p4), avg_depth, face_normal_3d) for external depth sorting and shading.
    
    Note: segments is automatically reduced for short cylinders to improve performance.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-10:
        return []
    
    # Unit direction vector
    ux, uy, uz = dx/length, dy/length, dz/length
    
    # Generate two perpendicular vectors for the circle base
    # Find a vector not parallel to the direction
    if abs(ux) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)
    
    # First perpendicular vector (cross product)
    v1x = ref[1]*uz - ref[2]*uy
    v1y = ref[2]*ux - ref[0]*uz
    v1z = ref[0]*uy - ref[1]*ux
    mag1 = math.sqrt(v1x*v1x + v1y*v1y + v1z*v1z)
    if mag1 < 1e-10:
        return []
    v1x, v1y, v1z = v1x/mag1, v1y/mag1, v1z/mag1
    
    # Second perpendicular vector (cross product of direction and v1)
    v2x = uy*v1z - uz*v1y
    v2y = uz*v1x - ux*v1z
    v2z = ux*v1y - uy*v1x
    mag2 = math.sqrt(v2x*v2x + v2y*v2y + v2z*v2z)
    if mag2 < 1e-10:
        return []
    v2x, v2y, v2z = v2x/mag2, v2y/mag2, v2z/mag2
    
    # Generate cylinder faces (side only, no caps for performance)
    # Use perspective-correct radii: each end gets its own scale factor
    camera_distance = CONFIG['projection'].get('camera_distance', 15.0)
    fov = CONFIG['projection'].get('fov', 500.0)
    
    # Compute scale factors for start and end based on depth
    z_start = start[2] + camera_distance
    z_end = end[2] + camera_distance
    z_start = max(z_start, 0.1)  # Avoid division by zero
    z_end = max(z_end, 0.1)
    scale_start = fov / z_start
    scale_end = fov / z_end
    
    faces = []
    for i in range(segments):
        angle1 = 2.0 * math.pi * i / segments
        angle2 = 2.0 * math.pi * (i + 1) / segments
        
        cos1, sin1 = math.cos(angle1), math.sin(angle1)
        cos2, sin2 = math.cos(angle2), math.sin(angle2)
        
        # Base circle points at start (with perspective-correct radius)
        # The radius needs to be scaled inversely to depth to appear constant on screen
        r_start = radius * scale_start / fov  # radius / scale = world units
        x1s = start[0] + v1x * r_start * cos1 + v2x * r_start * sin1
        y1s = start[1] + v1y * r_start * cos1 + v2y * r_start * sin1
        z1s = start[2] + v1z * r_start * cos1 + v2z * r_start * sin1
        
        x2s = start[0] + v1x * r_start * cos2 + v2x * r_start * sin2
        y2s = start[1] + v1y * r_start * cos2 + v2y * r_start * sin2
        z2s = start[2] + v1z * r_start * cos2 + v2z * r_start * sin2
        
        # Base circle points at end (with perspective-correct radius)
        r_end = radius * scale_end / fov
        x1e = end[0] + v1x * r_end * cos1 + v2x * r_end * sin1
        y1e = end[1] + v1y * r_end * cos1 + v2y * r_end * sin1
        z1e = end[2] + v1z * r_end * cos1 + v2z * r_end * sin1
        
        x2e = end[0] + v1x * r_end * cos2 + v2x * r_end * sin2
        y2e = end[1] + v1y * r_end * cos2 + v2y * r_end * sin2
        z2e = end[2] + v1z * r_end * cos2 + v2z * r_end * sin2
        
        # Project to screen (use provided width/height or fallback to defaults)
        w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
        h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
        p1s = project_3d_to_screen(x1s, y1s, z1s, w, h)
        p2s = project_3d_to_screen(x2s, y2s, z2s, w, h)
        p1e = project_3d_to_screen(x1e, y1e, z1e, w, h)
        p2e = project_3d_to_screen(x2e, y2e, z2e, w, h)
        
        if any(p is None for p in [p1s, p2s, p1e, p2e]):
            continue
        
        # Average depth for sorting (higher = closer)
        avg_depth = (z1s + z2s + z1e + z2e) / 4.0 + depth_offset
        
        # Compute face normal (cross product of two edges, normalized)
        # Edge vectors in 3D
        e1x = x2s - x1s
        e1y = y2s - y1s
        e1z = z2s - z1s
        e2x = x1e - x1s
        e2y = y1e - y1s
        e2z = z1e - z1s
        
        # Cross product e1 x e2
        nx = e1y*e2z - e1z*e2y
        ny = e1z*e2x - e1x*e2z
        nz = e1x*e2y - e1y*e2x
        n_mag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if n_mag > 1e-10:
            nx, ny, nz = nx/n_mag, ny/n_mag, nz/n_mag
        else:
            nx, ny, nz = ux, uy, uz  # Fallback to direction
        
        faces.append(((p1s, p2s, p2e, p1e), avg_depth, (nx, ny, nz)))
    
    return faces


def draw_3d_cone(apex, base_center, base_radius, segments, screen, color, depth_offset=0.0, width=None, height=None):
    """Draw a 3D cone as filled polygons, returns faces for depth sorting.
    
    Cone apex is the tip, base_center is the center of the circular base.
    Returns list of faces as ((p1, p2, apex_screen), avg_depth, face_normal_3d) for external depth sorting and shading.
    
    Note: segments is automatically reduced for small cones to improve performance.
    """
    dx = base_center[0] - apex[0]
    dy = base_center[1] - apex[1]
    dz = base_center[2] - apex[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-10:
        return []
    
    # Unit direction from apex to base
    ux, uy, uz = dx/length, dy/length, dz/length
    
    # Generate perpendicular basis for base circle
    if abs(ux) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)
    
    v1x = ref[1]*uz - ref[2]*uy
    v1y = ref[2]*ux - ref[0]*uz
    v1z = ref[0]*uy - ref[1]*ux
    mag1 = math.sqrt(v1x*v1x + v1y*v1y + v1z*v1z)
    if mag1 < 1e-10:
        return []
    v1x, v1y, v1z = v1x/mag1, v1y/mag1, v1z/mag1
    
    v2x = uy*v1z - uz*v1y
    v2y = uz*v1x - ux*v1z
    v2z = ux*v1y - uy*v1x
    mag2 = math.sqrt(v2x*v2x + v2y*v2y + v2z*v2z)
    if mag2 < 1e-10:
        return []
    v2x, v2y, v2z = v2x/mag2, v2y/mag2, v2z/mag2
    
    # Project apex to screen (use provided width/height or fallback to defaults)
    w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
    h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
    apex_screen = project_3d_to_screen(apex[0], apex[1], apex[2], w, h)
    if apex_screen is None:
        return []
    
    # Get perspective scale factors
    camera_distance = CONFIG['projection'].get('camera_distance', 15.0)
    fov = CONFIG['projection'].get('fov', 500.0)
    z_base = base_center[2] + camera_distance
    z_base = max(z_base, 0.1)
    base_scale = fov / z_base
    
    # Perspective-correct base radius
    corrected_base_radius = base_radius * base_scale / fov
    
    faces = []
    for i in range(segments):
        angle1 = 2.0 * math.pi * i / segments
        angle2 = 2.0 * math.pi * (i + 1) / segments
        
        cos1, sin1 = math.cos(angle1), math.sin(angle1)
        cos2, sin2 = math.cos(angle2), math.sin(angle2)
        
        # Two points on base circle (with perspective-correct radius)
        bx1 = base_center[0] + v1x * corrected_base_radius * cos1 + v2x * corrected_base_radius * sin1
        by1 = base_center[1] + v1y * corrected_base_radius * cos1 + v2y * corrected_base_radius * sin1
        bz1 = base_center[2] + v1z * corrected_base_radius * cos1 + v2z * corrected_base_radius * sin1
        
        bx2 = base_center[0] + v1x * corrected_base_radius * cos2 + v2x * corrected_base_radius * sin2
        by2 = base_center[1] + v1y * corrected_base_radius * cos2 + v2y * corrected_base_radius * sin2
        bz2 = base_center[2] + v1z * corrected_base_radius * cos2 + v2z * corrected_base_radius * sin2
        
        w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
        h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
        p1 = project_3d_to_screen(bx1, by1, bz1, w, h)
        p2 = project_3d_to_screen(bx2, by2, bz2, w, h)
        
        if p1 is None or p2 is None:
            continue
        
        # Average depth for sorting
        avg_depth = (bz1 + base_center[2]) / 2.0 + depth_offset
        
        # Compute face normal for the triangle (apex -> p1 -> p2)
        # Edge vectors in 3D
        e1x = bx1 - apex[0]
        e1y = by1 - apex[1]
        e1z = bz1 - apex[2]
        e2x = bx2 - apex[0]
        e2y = by2 - apex[1]
        e2z = bz2 - apex[2]
        
        # Cross product e1 x e2
        nx = e1y*e2z - e1z*e2y
        ny = e1z*e2x - e1x*e2z
        nz = e1x*e2y - e1y*e2x
        n_mag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if n_mag > 1e-10:
            nx, ny, nz = nx/n_mag, ny/n_mag, nz/n_mag
        else:
            nx, ny, nz = ux, uy, uz  # Fallback to direction
        
        faces.append(((apex_screen, p1, p2), avg_depth, (nx, ny, nz)))
    
    return faces


# ==================== 3D VECTOR RENDERING ====================

def draw_3d_vector_omega_drawables(center, direction, magnitude, config, geom_config, max_speed, screen, width=None, height=None):
    """Create drawables for the omega vector with alpha support.
    
    Returns list of drawables as (depth, 'polygon', face, color, normal, alpha) tuples,
    or empty list if magnitude is too small.
    
    Performance: Uses dynamic segment scaling - short vectors use fewer segments.
    """
    # Get geometry config FIRST - need tip_length for early exit checks
    tip_length = geom_config.get('tip_length', 2.0)  # Default to 2.0 (config value), not 18.0
    
    # Early-exit threshold from global config
    vr = CONFIG.get('vector_rendering', {})
    min_draw_ratio = vr.get('min_draw_ratio', 0.3)
    shaft_radius = geom_config.get('shaft_radius', 3.0)
    shaft_segments = geom_config.get('shaft_segments', 8)
    arrowhead_length_ratio = geom_config.get('arrowhead_length_ratio', 0.2)
    arrowhead_radius_ratio = geom_config.get('arrowhead_radius_ratio', 0.4)
    
    # Total arrow length (compute early for use in checks)
    total_length = min(magnitude * 0.5 + 20, 120)
    
    # Skip rendering if total length is below configurable threshold (invisible anyway)
    # This must happen BEFORE any drawable creation
    if total_length < tip_length * min_draw_ratio:
        return [], None
    
    shaft_length = total_length * (1.0 - arrowhead_length_ratio)
    arrowhead_base_radius = shaft_radius * arrowhead_radius_ratio
    
    # End point of shaft (before arrowhead)
    shaft_end = (
        center[0] + direction[0] * shaft_length,
        center[1] + direction[1] * shaft_length,
        center[2] + direction[2] * shaft_length
    )
    
    # Apex of cone (tip of arrow)
    cone_apex = (
        center[0] + direction[0] * total_length,
        center[1] + direction[1] * total_length,
        center[2] + direction[2] * total_length
    )
    
    # Get color from magnitude
    mag_norm = min(magnitude / max_speed, 1.0)
    r = int(80 + mag_norm * 175)
    g = int(80 + (1 - mag_norm) * 100)
    b = int(255 - mag_norm * 150)
    color = (r, g, b)
    
    # Compute alpha based on total_length vs tip_length
    alpha = compute_vector_alpha(total_length, tip_length)
    
    w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
    h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
    
    # Use dynamic segment scaling for performance
    dynamic_segs, _ = compute_dynamic_segments(shaft_segments, center, shaft_end, screen_width=w, screen_height=h)
    cylinder_faces = draw_3d_cylinder(center, shaft_end, shaft_radius, dynamic_segs, screen, color, depth_offset=0.01, width=w, height=h)
    
    dynamic_segs_cone, _ = compute_dynamic_segments(shaft_segments, shaft_end, cone_apex, screen_width=w, screen_height=h)
    cone_faces = draw_3d_cone(cone_apex, shaft_end, arrowhead_base_radius, dynamic_segs_cone, screen, color, depth_offset=-0.01, width=w, height=h)
    
    drawables = []
    for face, depth, normal in cylinder_faces:
        drawables.append((depth, 'polygon', face, color, normal, alpha))
    for face, depth, normal in cone_faces:
        drawables.append((depth, 'polygon', face, color, normal, alpha))
    
    return drawables, color


def draw_3d_vector(screen, start_3d, direction_3d, length, color, geom_config, proj_multiplier_x=0.15, proj_multiplier_y=0.12, width=None, height=None):
    """Draw a vector as a 3D cylinder with cone arrowhead, projecting through perspective.
    
    For tangential/centripetal vectors that start at cube vertices.
    Returns list of ((p1,p2,p3,...), depth) tuples for depth sorting.
    """
    if length < 1e-10:
        return []
    
    # Get geometry config
    shaft_radius = geom_config.get('shaft_radius', 2.0)
    shaft_segments = geom_config.get('shaft_segments', 6)
    tip_length = geom_config.get('tip_length', 14.0)  # Fixed arrowhead length (world units)
    arrowhead_radius_ratio = geom_config.get('arrowhead_radius_ratio', 5)
    
    # End point in 3D (direction is already normalized)
    end_3d = (
        start_3d[0] + direction_3d[0] * length,
        start_3d[1] + direction_3d[1] * length,
        start_3d[2] + direction_3d[2] * length
    )
    
    w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
    h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
    p_start = project_3d_to_screen(start_3d[0], start_3d[1], start_3d[2], w, h)
    p_end = project_3d_to_screen(end_3d[0], end_3d[1], end_3d[2], w, h)
    if p_start is None or p_end is None:
        return []
    
    shaft_length_3d = max(length - tip_length, 0.01)  # Shaft = total minus fixed tip length
    arrowhead_base_radius = shaft_radius * arrowhead_radius_ratio
    
    # Shaft end point (where cone base begins)
    shaft_end_3d = (
        start_3d[0] + direction_3d[0] * shaft_length_3d,
        start_3d[1] + direction_3d[1] * shaft_length_3d,
        start_3d[2] + direction_3d[2] * shaft_length_3d
    )
    
    w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
    h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
    cylinder_faces = draw_3d_cylinder(start_3d, shaft_end_3d, shaft_radius, shaft_segments, screen, color, depth_offset=0.01, width=w, height=h)
    cone_apex = (
        start_3d[0] + direction_3d[0] * length,
        start_3d[1] + direction_3d[1] * length,
        start_3d[2] + direction_3d[2] * length
    )
    cone_faces = draw_3d_cone(cone_apex, shaft_end_3d, arrowhead_base_radius, shaft_segments, screen, color, depth_offset=-0.01, width=w, height=h)
    
    return cylinder_faces + cone_faces


def draw_3d_vector_omega_on_screen(screen, center_screen, direction_screen, total_omega, config, geom_config):
    """Draw the omega vector as a 3D cylinder with cone arrowhead in screen-projected space.
    
    This handles the omega vector which originates from the cube center projected to screen.
    direction_screen: the direction in screen space (already rotated by camera)
    """
    if total_omega < 0.1:
        return
    
    # Get geometry config
    shaft_radius = geom_config.get('shaft_radius', 3.0)
    shaft_segments = geom_config.get('shaft_segments', 8)
    arrowhead_length_ratio = geom_config.get('arrowhead_length_ratio', 0.2)
    arrowhead_radius_ratio = geom_config.get('arrowhead_radius_ratio', 0.4)
    
    # Scale length with magnitude
    total_length = min(total_omega * 0.5 + 20, 120)
    shaft_length = total_length * (1.0 - arrowhead_length_ratio)
    arrowhead_length = total_length * arrowhead_length_ratio
    arrowhead_base_radius = shaft_radius * arrowhead_radius_ratio
    
    # Shaft end and cone apex in screen space
    shaft_end_screen = (
        center_screen[0] + direction_screen[0] * shaft_length,
        center_screen[1] + direction_screen[1] * shaft_length
    )
    cone_apex_screen = (
        center_screen[0] + direction_screen[0] * total_length,
        center_screen[1] + direction_screen[1] * total_length
    )
    
    # Get color from magnitude
    mag_norm = min(total_omega / config['angular_velocity']['max_speed'], 1.0)
    r = int(80 + mag_norm * 175)
    g = int(80 + (1 - mag_norm) * 100)
    b = int(255 - mag_norm * 150)
    color = (r, g, b)
    
    # Draw cylinder shaft as filled polygon strip in screen space
    # Draw cylinder shaft as a thick line (polygon) in screen space
    dx = direction_screen[0]
    dy = direction_screen[1]
    # Perpendicular in screen space
    px = -dy
    py = dx
    mag = math.sqrt(px*px + py*py)
    if mag > 1e-10:
        px, py = px/mag * shaft_radius, py/mag * shaft_radius
        
        # Draw shaft as filled polygon strip
        n = shaft_segments
        shaft_faces = []
        for i in range(n):
            angle1 = 2.0 * math.pi * i / n
            angle2 = 2.0 * math.pi * (i + 1) / n
            cos1, sin1 = math.cos(angle1), math.sin(angle1)
            cos2, sin2 = math.cos(angle2), math.sin(angle2)
            
            # At shaft start
            x1s = center_screen[0] + px * cos1
            y1s = center_screen[1] + py * sin1
            x2s = center_screen[0] + px * cos2
            y2s = center_screen[1] + py * sin2
            
            # At shaft end
            x1e = shaft_end_screen[0] + px * cos1
            y1e = shaft_end_screen[1] + py * sin1
            x2e = shaft_end_screen[0] + px * cos2
            y2e = shaft_end_screen[1] + py * sin2
            
            shaft_faces.append(((int(x1s), int(y1s)), (int(x2s), int(y2s)), (int(x2e), int(y2e)), (int(x1e), int(y1e))))
        
        # Draw a red circle at the base of the shaft (before shaft faces so it appears underneath)
        pygame.draw.circle(screen, (255, 0, 0), (int(center_screen[0]), int(center_screen[1])), int(shaft_radius))
        
        for face in shaft_faces:
            pygame.draw.polygon(screen, color, face, 0)  # filled
        
        # Draw cone arrowhead as filled polygon
        # Base of cone at shaft_end_screen
        for i in range(n):
            angle1 = 2.0 * math.pi * i / n
            angle2 = 2.0 * math.pi * (i + 1) / n
            cos1, sin1 = math.cos(angle1), math.sin(angle1)
            cos2, sin2 = math.cos(angle2), math.sin(angle2)
            
            bx1 = shaft_end_screen[0] + px * cos1 * arrowhead_base_radius / shaft_radius
            by1 = shaft_end_screen[1] + py * sin1 * arrowhead_base_radius / shaft_radius
            bx2 = shaft_end_screen[0] + px * cos2 * arrowhead_base_radius / shaft_radius
            by2 = shaft_end_screen[1] + py * sin2 * arrowhead_base_radius / shaft_radius
            
            pygame.draw.polygon(screen, color, [(int(cone_apex_screen[0]), int(cone_apex_screen[1])), (int(bx1), int(by1)), (int(bx2), int(by2))], 0)


# ==================== 3D FACE NORMALS ====================

def compute_face_normal_from_verts(face_indices, verts):
    """Compute the outward normal of a face from its rotated vertex positions.
    
    Uses the cross product of two edge vectors to get the face normal.
    """
    p0 = np.array(verts[face_indices[0]])
    p1 = np.array(verts[face_indices[1]])
    p2 = np.array(verts[face_indices[2]])
    
    # Two edge vectors
    e1 = p1 - p0
    e2 = p2 - p0
    
    # Cross product gives the face normal
    normal = np.cross(e1, e2)
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal = normal / norm
    return normal


def draw_3d_face_normals(verts, screen, width=None, height=None):
    """Draw normal vectors from the center of each cube face as 3D cylinder+cone objects.
    
    Draws normals on all 6 faces using CUBE_FACES_3D from geometry.
    Normals follow the cube's rotation (computed from rotated vertex positions).
    Returns list of drawables for depth sorting.
    """
    from geometry import CUBE_FACES_3D
    
    # Get normal length and colors from unified vectors config
    vectors_cfg = CONFIG.get('vectors', {})
    face_cfg = vectors_cfg.get('face_normals', {})
    normal_face_len = face_cfg.get('fixed_length', 3)
    
    # Get face normals color from config (no secondary color for face normals)
    face_color = tuple(face_cfg.get('color', [200, 200, 255]))
    
    # Get geometry config for normal arrows
    normal_geom = face_cfg.get('geometry', {})
    shaft_radius = normal_geom.get('shaft_radius', 1.5)
    shaft_segments = normal_geom.get('shaft_segments', 6)
    tip_length = normal_geom.get('tip_length', 1.5)
    arrowhead_radius_ratio = normal_geom.get('arrowhead_radius_ratio', 5)
    
    drawables = []
    
    for face_info in CUBE_FACES_3D:
        face_indices = face_info['indices']
        # Face normals use a single color (no secondary)
        color = face_color
        
        # Compute face center from rotated vertices
        face_verts_list = [verts[i] for i in face_indices]
        center_pt = np.mean(face_verts_list, axis=0)
        
        # Compute normal from rotated face geometry (follows cube rotation)
        face_normal = compute_face_normal_from_verts(face_indices, verts)
        
        # Compute tip point (extend normal from face center)
        tip_3d = tuple(center_pt[i] + face_normal[i] * normal_face_len for i in range(3))
        
        w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
        h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
        p_start = project_3d_to_screen(center_pt[0], center_pt[1], center_pt[2], w, h)
        p_tip = project_3d_to_screen(tip_3d[0], tip_3d[1], tip_3d[2], w, h)
        if p_start is None or p_tip is None:
            continue
        
        # Compute shaft length (total minus fixed tip)
        dx = tip_3d[0] - center_pt[0]
        dy = tip_3d[1] - center_pt[1]
        dz = tip_3d[2] - center_pt[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < tip_length:
            continue
        
        shaft_len = max(length - tip_length, 0.01)
        shaft_end_3d = (
            center_pt[0] + dx/length * shaft_len,
            center_pt[1] + dy/length * shaft_len,
            center_pt[2] + dz/length * shaft_len
        )
        cone_apex_3d = tip_3d
        arrowhead_base_r = shaft_radius * arrowhead_radius_ratio
        
        w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
        h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
        cylinder_faces = draw_3d_cylinder(center_pt, shaft_end_3d, shaft_radius, shaft_segments, screen, color, depth_offset=0.01, width=w, height=h)
        cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_r, shaft_segments, screen, color, depth_offset=-0.01, width=w, height=h)
        
        # Average depth for sorting
        avg_depth = (center_pt[2] + shaft_end_3d[2] + cone_apex_3d[2]) / 3.0
        
        # Compute alpha for face normals based on length vs tip length
        n_alpha = compute_vector_alpha(length, tip_length)
        
        for face, depth, normal in cylinder_faces:
            drawables.append((avg_depth, 'polygon', face, color, normal, n_alpha))
        for face, depth, normal in cone_faces:
            drawables.append((avg_depth, 'polygon', face, color, normal, n_alpha))
    
    return drawables


# ==================== 3D VELOCITY VECTORS ====================

def draw_3d_velocity_vectors(verts, total_omega_mag, screen, omega_x=0, omega_y=0, omega_z=0, max_vectors=3, show_tangential=True, show_centripetal=True, view_x=0, view_y=0, view_z=0, vertex_mask=None, width=None, height=None):
    """Draw tangential velocity vectors and centripetal acceleration vectors as 3D cylinder+cone objects.
    
    Returns two lists: (tangential_drawables, centripetal_drawables)
    Each drawable is ((depth, 'polygon', screen_coords_tuple, color)).
    Vectors scale linearly from min_length to max_length based on normalized omega magnitude.
    """
    # Get geometry configs from unified 'vectors' section
    vectors_cfg = CONFIG.get('vectors', {})
    tang_geom = vectors_cfg.get('tangential_velocity', {}).get('geometry', {})
    cent_geom = vectors_cfg.get('centripetal_acceleration', {}).get('geometry', {})
    
    # Get vector config from unified 'vectors' section
    tang_cfg = vectors_cfg.get('tangential_velocity', {})
    cent_cfg = vectors_cfg.get('centripetal_acceleration', {})
    
    # Get per-vector colors from config (fallback to legacy COLORS dict)
    tang_color = tuple(tang_cfg.get('color', CONFIG['colors'].get('velocity', [140, 190, 255])))
    tang_color_secondary = tuple(tang_cfg.get('color_secondary', [100, 255, 200]))
    cent_color = tuple(cent_cfg.get('color', CONFIG['colors'].get('highlight', [255, 220, 180])))
    cent_color_secondary = tuple(cent_cfg.get('color_secondary', [255, 100, 100]))
    
    tang_scale = tang_cfg.get('scale_factor', 0.2)
    tang_min_length = tang_cfg.get('min_length', 0)
    tang_clamp_max = tang_cfg.get('clamp_max', 25)
    cent_scale = cent_cfg.get('scale_factor', 0.05)
    cent_min_length = cent_cfg.get('min_length', 0)
    cent_clamp_max = cent_cfg.get('clamp_max', 25)
    
    # Use full omega vector for proper cross product
    omega_vec = np.array([omega_x, omega_y, omega_z])
    
    # Select only a subset of vertices
    valid_verts = []
    for i, v in enumerate(verts):
        r_vec = np.array(v)
        r_mag = math.sqrt(sum(x**2 for x in r_vec))
        if r_mag >= 0.5:
            # Apply vertex mask if provided
            if vertex_mask is not None:
                if not vertex_mask.get(i, False):
                    continue
            valid_verts.append((i, v))
    
    selected_verts = [(i, v) for i, v in valid_verts[:max_vectors]]
    
    tangential_drawables = []
    centripetal_drawables = []
    
    for vert_idx, v in selected_verts:
        r_vec = np.array(v)
        
        # Compute tangential velocity via cross product: v = omega x r
        tangential_v = cross_product(omega_vec, r_vec)
        
        # Draw tangential velocity if enabled
        if show_tangential:
            speed = math.sqrt(sum(c**2 for c in tangential_v))
            
            # Scale by speed using scale_factor, then clamp
            display_length = speed * tang_scale
            display_length = min(display_length, tang_clamp_max)
            display_length = max(display_length, tang_min_length)
            
            # Compute blend factor based on display length ratio
            # Blending starts at tip_length (100% opacity) and ends at clamp_max
            t_tip_length = tang_geom.get('tip_length', 1.5)
            if tang_clamp_max > t_tip_length:
                tang_blend = (display_length - t_tip_length) / (tang_clamp_max - t_tip_length)
            else:
                tang_blend = 0.0
            tang_blend = max(0.0, min(1.0, tang_blend))
            tang_color_final = (
                int(tang_color[0] * (1 - tang_blend) + tang_color_secondary[0] * tang_blend),
                int(tang_color[1] * (1 - tang_blend) + tang_color_secondary[1] * tang_blend),
                int(tang_color[2] * (1 - tang_blend) + tang_color_secondary[2] * tang_blend),
            )
            
            # Draw even when below tip length - alpha handles visibility
            t_tip_length = tang_geom.get('tip_length', 1.5)
            
            # Early exit: skip rendering if display length is below configurable threshold (invisible anyway)
            vr = CONFIG.get('vector_rendering', {})
            min_draw_ratio = vr.get('min_draw_ratio', 0.3)
            if display_length < t_tip_length * min_draw_ratio:
                continue
            
            if speed >= 0.01:
                scaled_v = (tangential_v[0] / speed * display_length,
                            tangential_v[1] / speed * display_length,
                            tangential_v[2] / speed * display_length)
                
                # Get geometry config for tangential vectors
                t_shaft_radius = tang_geom.get('shaft_radius', 2.0)
                t_shaft_segments = tang_geom.get('shaft_segments', 6)
                t_arrowhead_ratio = tang_geom.get('arrowhead_length_ratio', 0.25)
                t_arrowhead_radius_ratio = tang_geom.get('arrowhead_radius_ratio', 0.35)
                
                # Compute 3D positions - scaled_v is already the visualization-scaled displacement
                start_3d = (v[0], v[1], v[2])
                
                # End point in world space (use scaled_v directly as displacement)
                end_world = (
                    start_3d[0] + scaled_v[0],
                    start_3d[1] + scaled_v[1],
                    start_3d[2] + scaled_v[2]
                )
                
                # Rotate both points by camera angles for projection
                start_cam = start_3d  # Already rotated in driver
                end_cam = end_world  # Already rotated in driver
                
                w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
                h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
                p_start = project_3d_to_screen(start_cam[0], start_cam[1], start_cam[2], w, h)
                p_end = project_3d_to_screen(end_cam[0], end_cam[1], end_cam[2], w, h)
                if p_start is None or p_end is None:
                    continue
                
                # Shaft length: total minus fixed arrowhead length (tip_length from config)
                tip_length = tang_geom.get('tip_length', 1.5)
                arrowhead_base_r = t_shaft_radius * t_arrowhead_radius_ratio
                
                # Compute direction vector in camera-rotated space
                dx = end_cam[0] - start_cam[0]
                dy = end_cam[1] - start_cam[1]
                dz = end_cam[2] - start_cam[2]
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                if length < 1e-10:
                    continue
                
                # Shaft is total vector minus the fixed arrowhead tip (fixed length)
                shaft_len = max(length - tip_length, 0.01)
                shaft_end_3d = (
                    start_cam[0] + dx/length * shaft_len,
                    start_cam[1] + dy/length * shaft_len,
                    start_cam[2] + dz/length * shaft_len
                )
                cone_apex_3d = end_cam
                
                # Average depth for sorting (use camera-rotated z-depth)
                avg_depth = (start_cam[2] + shaft_end_3d[2] + cone_apex_3d[2]) / 3.0
                
                w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
                h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
                cylinder_faces = draw_3d_cylinder(start_cam, shaft_end_3d, t_shaft_radius, t_shaft_segments, screen, tang_color_final, depth_offset=0.01, width=w, height=h)
                cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_r, t_shaft_segments, screen, tang_color_final, depth_offset=-0.01, width=w, height=h)
                
                # Compute alpha based on display length vs tip length
                t_alpha = compute_vector_alpha(display_length, t_tip_length)
                
                for face, depth, normal in cylinder_faces:
                    tangential_drawables.append((avg_depth, 'polygon', face, tang_color_final, normal, t_alpha))
                for face, depth, normal in cone_faces:
                    tangential_drawables.append((avg_depth, 'polygon', face, tang_color_final, normal, t_alpha))
        
        # Draw centripetal acceleration if enabled
        if show_centripetal:
            centripetal_a = cross_product(omega_vec, tangential_v)
            a_mag = math.sqrt(sum(c**2 for c in centripetal_a))
            
            # Scale by magnitude using scale_factor, then clamp
            display_length = a_mag * cent_scale
            display_length = min(display_length, cent_clamp_max)
            display_length = max(display_length, cent_min_length)
            
            # Compute blend factor based on display length ratio
            # Blending starts at tip_length (100% opacity) and ends at clamp_max
            c_tip_length = cent_geom.get('tip_length', 2.0)
            if cent_clamp_max > c_tip_length:
                cent_blend = (display_length - c_tip_length) / (cent_clamp_max - c_tip_length)
            else:
                cent_blend = 0.0
            cent_blend = max(0.0, min(1.0, cent_blend))
            cent_color_final = (
                int(cent_color[0] * (1 - cent_blend) + cent_color_secondary[0] * cent_blend),
                int(cent_color[1] * (1 - cent_blend) + cent_color_secondary[1] * cent_blend),
                int(cent_color[2] * (1 - cent_blend) + cent_color_secondary[2] * cent_blend),
            )
            
            # Draw even when below tip length - alpha handles visibility
            c_tip_length = cent_geom.get('tip_length', 2.0)
            
            # Early exit: skip rendering if display length is below configurable threshold (invisible anyway)
            vr = CONFIG.get('vector_rendering', {})
            min_draw_ratio = vr.get('min_draw_ratio', 0.3)
            if display_length < c_tip_length * min_draw_ratio:
                continue
            
            if a_mag >= 0.01:
                a_unit = np.array([c / a_mag for c in centripetal_a])
                scaled_a = tuple(a_unit[i] * display_length for i in range(3))
                
                # Get geometry config for centripetal vectors
                c_shaft_radius = cent_geom.get('shaft_radius', 1.5)
                c_shaft_segments = cent_geom.get('shaft_segments', 6)
                tip_length_c = cent_geom.get('tip_length', 2.0)
                arrowhead_base_r_c = c_shaft_radius * cent_geom.get('arrowhead_radius_ratio', 5)
                
                # Compute 3D positions - scaled_a is already the visualization-scaled displacement
                start_3d = (v[0], v[1], v[2])
                
                # End point in world space (use scaled_a directly as displacement)
                end_world = (
                    start_3d[0] + scaled_a[0],
                    start_3d[1] + scaled_a[1],
                    start_3d[2] + scaled_a[2]
                )
                
                # Rotate both points by camera angles for projection
                start_cam = start_3d  # Already rotated in driver
                end_cam = end_world  # Already rotated in driver
                
                w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
                h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
                p_start = project_3d_to_screen(start_cam[0], start_cam[1], start_cam[2], w, h)
                p_end = project_3d_to_screen(end_cam[0], end_cam[1], end_cam[2], w, h)
                if p_start is None or p_end is None:
                    continue
                
                # Shaft length: total minus fixed arrowhead tip (tip_length from config)
                dx = end_cam[0] - start_cam[0]
                dy = end_cam[1] - start_cam[1]
                dz = end_cam[2] - start_cam[2]
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                if length < 1e-10:
                    continue
                
                # Shaft is total vector minus the fixed arrowhead tip (fixed length)
                shaft_len_c = max(length - tip_length_c, 0.01)
                shaft_end_3d = (
                    start_cam[0] + dx/length * shaft_len_c,
                    start_cam[1] + dy/length * shaft_len_c,
                    start_cam[2] + dz/length * shaft_len_c
                )
                cone_apex_3d = end_cam
                
                # Average depth for sorting (use camera-rotated z-depth)
                avg_depth = (start_cam[2] + shaft_end_3d[2] + cone_apex_3d[2]) / 3.0
                
                w = width if width is not None else (screen.get_width() if screen is not None else SCREEN_W)
                h = height if height is not None else (screen.get_height() if screen is not None else SCREEN_H)
                cylinder_faces = draw_3d_cylinder(start_cam, shaft_end_3d, c_shaft_radius, c_shaft_segments, screen, cent_color_final, depth_offset=0.01, width=w, height=h)
                cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_r_c, c_shaft_segments, screen, cent_color_final, depth_offset=-0.01, width=w, height=h)
                
                # Compute alpha based on display length vs tip length
                c_alpha = compute_vector_alpha(display_length, c_tip_length)
                
                for face, depth, normal in cylinder_faces:
                    centripetal_drawables.append((avg_depth, 'polygon', face, cent_color_final, normal, c_alpha))
                for face, depth, normal in cone_faces:
                    centripetal_drawables.append((avg_depth, 'polygon', face, cent_color_final, normal, c_alpha))
    
    return tangential_drawables, centripetal_drawables
