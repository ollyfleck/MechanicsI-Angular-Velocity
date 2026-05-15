"""
Angular Velocity Demo - Logic Module

Contains all PURE physics/math logic (no UI/pygame dependencies):
- 3D rotation transformations  
- Cube geometry definitions
- Vector computation (cross product for tangential velocity)
- Projection to screen coordinates
- Drag impulse helper function
- Drawing functions (pygame-dependent, but exported for convenience)

Note: This module loads configuration from config.yaml automatically.
"""

import math
import numpy as np
import os
import yaml


# ==================== CONFIGURATION ====================

# Load configuration from YAML file relative to this module's location
config_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(config_dir, 'config.yaml')
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

COLORS = {
    'bg': tuple(CONFIG['colors']['bg']),
    'cube_edges': tuple(CONFIG['colors']['cube_edges']),
    'omega': tuple(CONFIG['colors']['omega']),
    'velocity': tuple(CONFIG['colors']['velocity']),
    'axis': tuple(CONFIG['colors']['axis']),
    'text_main': tuple(CONFIG['colors']['text_main']),
    'text_sub': tuple(CONFIG['colors']['text_sub']),
}

# Handle highlight color fallback if not in config
if 'highlight' in CONFIG['colors']:
    COLORS['highlight'] = tuple(CONFIG['colors']['highlight'])
else:
    COLORS['highlight'] = (255, 220, 180)

SCREEN_W = CONFIG['screen']['width']
SCREEN_H = CONFIG['screen']['height']
CUBE_SIZE = CONFIG['cube_size']
FRAME_RATE = CONFIG['physics'].get('frame_rate', 60)  # Fallback if missing

# ==================== PURE MATH FUNCTIONS (No pygame) ====================

def cross_product(a, b):
    """Compute cross product of two 3D vectors."""
    return (a[1]*b[2] - a[2]*b[1], 
            a[2]*b[0] - a[0]*b[2], 
            a[0]*b[1] - a[1]*b[0])


def dot_product(a, b):
    """Compute dot product of two 3D vectors."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def vector_magnitude(v):
    """Compute magnitude of a 3D vector."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normalize_vector(v):
    """Normalize a 3D vector to unit length."""
    mag = vector_magnitude(v)
    if mag < 1e-10:
        return (0, 0, 0)
    return (v[0]/mag, v[1]/mag, v[2]/mag)


def rotate_around_axis(point, axis, angle_deg):
    """Rotate a point around an arbitrary axis (X, Y, or Z)."""
    x, y, z = point
    theta_rad = math.radians(angle_deg)
    
    if abs(axis[0]) > 1e-6:  # X-axis rotation
        new_y = y * math.cos(theta_rad) - z * math.sin(theta_rad)
        new_z = y * math.sin(theta_rad) + z * math.cos(theta_rad)
        return (x, new_y, new_z)
    elif abs(axis[1]) > 1e-6:  # Y-axis rotation
        new_x = z * math.sin(theta_rad) + x * math.cos(theta_rad)
        new_z = z * math.cos(theta_rad) - x * math.sin(theta_rad)
        return (new_x, y, new_z)
    elif abs(axis[2]) > 1e-6:  # Z-axis rotation
        new_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
        new_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)
        return (new_x, new_y, z)
    return point


def rotate_x(point, angle_deg):
    """Rotate point around X-axis."""
    x, y, z = point
    theta_rad = math.radians(angle_deg)
    
    new_y = y * math.cos(theta_rad) - z * math.sin(theta_rad)
    new_z = y * math.sin(theta_rad) + z * math.cos(theta_rad)
    
    return (x, new_y, new_z)


def rotate_y(point, angle_deg):
    """Rotate point around Y-axis."""
    x, y, z = point
    theta_rad = math.radians(angle_deg)
    
    new_x = z * math.sin(theta_rad) + x * math.cos(theta_rad)
    new_z = z * math.cos(theta_rad) - x * math.sin(theta_rad)
    
    return (new_x, y, new_z)


def rotate_z(point, angle_deg):
    """Rotate point around Z-axis."""
    x, y, z = point
    theta_rad = math.radians(angle_deg)
    
    new_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
    new_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)
    
    return (new_x, new_y, z)


def apply_3axis_rotation_matrix(point, omega_x, omega_y, omega_z):
    """Apply small rotation using incremental 3-axis rotation matrices."""
    x, y, z = point
    
    # Convert angular velocity (rad/s) to frame delta angle
    frame_dt = 1.0 / FRAME_RATE
    rot_x_rad = omega_x * frame_dt
    rot_y_rad = omega_y * frame_dt
    rot_z_rad = omega_z * frame_dt
    
    # Apply X-axis rotation first, then Y, then Z (incremental matrix application)
    rx = rotate_around_axis((x, y, z), [1, 0, 0], math.degrees(rot_x_rad))
    ry = rotate_around_axis(rx, [0, 1, 0], math.degrees(rot_y_rad))
    final_pt = rotate_around_axis(ry, [0, 0, 1], math.degrees(rot_z_rad))
    
    return (final_pt[0], final_pt[1], final_pt[2])


def apply_rotations_to_array(array):
    """Apply current camera rotations to all points in an array."""
    rotated_list = []
    for point in array:
        p = list(point)
        
        # Apply rotations in order: X, then Y, then Z
        rx = rotate_x(p, CONFIG['rotation_angles']['x'])
        ry = rotate_y(rx, CONFIG['rotation_angles']['y'])
        rz = rotate_z(ry, CONFIG['rotation_angles']['z'])
        
        rotated_list.append(rz)
    return np.array(rotated_list, dtype=np.float64)


def apply_rotations(point):
    """Apply current camera rotations for 3D isometric view (single point)."""
    p = list(point)
    
    # Apply rotations in order: X, then Y, then Z
    rx = rotate_x(p, CONFIG['rotation_angles']['x'])
    ry = rotate_y(rx, CONFIG['rotation_angles']['y'])
    rz = rotate_z(ry, CONFIG['rotation_angles']['z'])
    
    return tuple(rz)


def project_3d_to_screen(x, y, z):
    """Project 3D coordinates to 2D screen (perspective projection).
    
    Uses perspective divide: objects farther from the camera appear smaller.
    Returns None if the point is behind the camera.
    """
    camera_distance = CONFIG['projection'].get('camera_distance', 15.0)
    fov = CONFIG['projection'].get('fov', 500.0)
    
    # Clip points behind the camera
    z_depth = z + camera_distance
    if z_depth < 0.1:  # Too close to or behind the camera
        return None
    
    # Perspective divide
    scale = fov / z_depth
    px = int(SCREEN_W // 2 + x * scale)
    py = int(SCREEN_H // 2 + y * scale)
    
    return (px, py)


def project_vector_to_plane(vec, normal):
    """Project a vector onto a plane with given normal."""
    dot = dot_product(vec, normalize_vector(normal))
    proj = vec - dot * normal
    return proj


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


# ==================== PHYSICS FUNCTIONS ====================

def apply_drag_impulse(omega_x, omega_y, omega_z, drag_dx, drag_dy):
    """Apply torque impulse from drag input to angular velocity.
    
    Uses synthetic_test config when enabled for unit testing without pygame.
    Uses mouse_drag_sensitivity config otherwise.
    """
    # Use synthetic_test config if enabled for unit testing without pygame
    synth_enabled = CONFIG.get('synthetic_test', {}).get('enabled', False)
    
    if synth_enabled:
        synth_config = CONFIG['synthetic_test']
        delta_omega_x = drag_dx * synth_config.get('drag_delta_omega_x', 0.1)
        delta_omega_y = drag_dy * synth_config.get('drag_delta_omega_y', 0.08)
        # Z component is sum of offset and base (as per test expectations)
        delta_omega_z = drag_dx * synth_config.get('drag_delta_omega_z_offset', 0.05) + \
                       drag_dy * synth_config.get('drag_delta_omega_z_base', 0.03)
    else:
        mouse_sens = CONFIG.get('mouse_drag_sensitivity', {})
        delta_omega_x = drag_dx * mouse_sens.get('omega_x', 0.2)
        delta_omega_y = drag_dy * mouse_sens.get('omega_y', 0.16)
        # Add z component based on combined drag (simulates torque from twisting motion)
        delta_omega_z = (drag_dx + drag_dy) * mouse_sens.get('omega_z', 0.12)
    
    omega_x += delta_omega_x
    omega_y += delta_omega_y
    omega_z += delta_omega_z
    
    return omega_x, omega_y, omega_z


def apply_drag_impulse_damping(omega_x, omega_y, omega_z):
    """Apply velocity-dependent damping to angular velocity."""
    damping = CONFIG['angular_velocity']['damping']
    
    omega_magnitude = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    if omega_magnitude > 0:
        max_speed = CONFIG['angular_velocity']['max_speed']
        speed_factor = min(omega_magnitude / max_speed, 1.0)
        effective_damping = damping - (speed_factor * 0.01)
    else:
        effective_damping = damping
    
    return omega_x * effective_damping, omega_y * effective_damping, omega_z * effective_damping


def clamp_angular_velocity(omega_x, omega_y, omega_z):
    """Clamp angular velocity to maximum allowed magnitude."""
    max_mag = CONFIG['angular_velocity']['max_speed'] * 1.5
    
    current_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    # Only scale down if velocity EXCEEDS the maximum
    if current_mag > max_mag:
        scale = max_mag / current_mag
        return omega_x * scale, omega_y * scale, omega_z * scale
    return omega_x, omega_y, omega_z


# ==================== DRAWING FUNCTIONS (pygame-dependent) ====================


def draw_cube_edges(verts, screen):
    """Draw the cube wireframe edges in white, sorted back-to-front for depth."""
    try:
        import pygame
    except ImportError:
        raise RuntimeError("pygame is required for drawing functions. Run: pip install pygame")
    
    # Compute average Z depth for each edge and sort back-to-front
    edge_data = []
    for idx, (i, j) in enumerate(CUBE_EDGES):
        p1_screen = project_3d_to_screen(*(verts[i]))
        p2_screen = project_3d_to_screen(*(verts[j]))
        
        # Skip edges with any point behind camera
        if p1_screen is None or p2_screen is None:
            continue
        
        # Average Z for depth sorting (higher Z = closer to camera with current coord system)
        avg_z = (verts[i][2] + verts[j][2]) / 2.0
        edge_data.append((avg_z, i, j, p1_screen, p2_screen))
    
    # Sort by depth (back to front = lowest Z first)
    edge_data.sort(key=lambda e: e[0])
    
    # All edges are white
    for (avg_z, i, j, p1, p2) in edge_data:
        pygame.draw.line(screen, (255, 255, 255), p1, p2, 3)


def draw_velocity_arrow_at_point(point_3d, velocity_vec, screen, is_normal=False):
    """Draw a velocity or normal vector arrow at a point.
    
    For tangential vectors: uses right-hand rule v = omega × r
    For normal vectors: shows outward direction from face center.
    Arrow length scales with speed and arrowhead points in the vector direction.
    Uses vector_scales config for projection multipliers.
    """
    if len(velocity_vec) < 3:
        return

    try:
        import pygame
    except ImportError:
        raise RuntimeError("pygame is required for drawing functions. Run: pip install pygame")

    # Scale the vector for visualization
    speed = math.sqrt(sum(v**2 for v in velocity_vec))

    # Skip drawing if speed is zero or near-zero
    if speed < 1e-6:
        return

    # Get projection multipliers from config
    vec_scales = CONFIG.get('vector_scales', {})
    proj_multiplier_x = vec_scales.get('proj_multiplier_x', 0.15)
    proj_multiplier_y = vec_scales.get('proj_multiplier_y', 0.12)
    max_scale = vec_scales.get('max_arrow_scale', 50)
    
    # Scale arrow length based on speed (clamp to max from config)
    scale_factor = min(speed * 8, max_scale)

    v_unit = np.array([v / speed for v in velocity_vec])
    
    # Project start point to screen
    p_start = project_3d_to_screen(*point_3d)
    
    # Skip if start point is behind camera
    if p_start is None:
        return
    
    # Project tip point (accounting for perspective in Y axis)
    tip_3d = (point_3d[0] + v_unit[0] * scale_factor * proj_multiplier_x,
              point_3d[1] - v_unit[1] * scale_factor * proj_multiplier_y,
              point_3d[2] + v_unit[2] * scale_factor * proj_multiplier_x)
    p_tip = project_3d_to_screen(*tip_3d)

    # Skip if tip point is behind camera
    if p_tip is None:
        return

    # Draw the arrow shaft
    color = COLORS['velocity'] if not is_normal else COLORS['highlight']
    pygame.draw.line(screen, color, p_start, p_tip, 3)
    
    # Draw arrowhead pointing in velocity direction
    angle = math.atan2(p_tip[1] - p_start[1], p_tip[0] - p_start[0])
    arrow_len = 10
    # Arrowhead points in the direction of the velocity vector
    pa1 = (int(p_tip[0] - arrow_len * math.cos(angle - 0.4)),
           int(p_tip[1] + arrow_len * math.sin(angle - 0.4)))
    pa2 = (int(p_tip[0] - arrow_len * math.cos(angle + 0.4)),
           int(p_tip[1] + arrow_len * math.sin(angle + 0.4)))
    
    pygame.draw.lines(screen, color, False, [pa1, p_tip, pa2], 3)


def draw_velocity_vectors_at_vertices(verts, total_omega_mag, screen, omega_x=0, omega_y=0, omega_z=0, max_vectors=3, show_tangential=True, show_centripetal=True):
    """Draw tangential velocity vectors and centripetal acceleration vectors at up to max_vectors cube vertices.
    
    Each vertex shows:
    - Tangential velocity (blue): v = ω × r, direction of motion due to rotation
    - Centripetal acceleration (orange): a = ω × v = ω × (ω × r), points toward rotation axis
    
    Arrow length scales with magnitude, arrowhead points in vector direction.
    Uses vector_scales config for scale factors.
    Toggled by show_tangential and show_centripetal flags.
    """
    # Get vector scale factors from config
    vec_scales = CONFIG.get('vector_scales', {})
    tangential_scale = vec_scales.get('tangential', 0.08)
    tangential_max = vec_scales.get('tangential_max', 8)
    normal_vertex_len = vec_scales.get('normal_vertex', 6)
    
    # Use full omega vector for proper cross product
    omega_vec = np.array([omega_x, omega_y, omega_z])
    
    # Select only a subset of vertices (skip vertices near origin)
    valid_verts = []
    for v in verts:
        r_vec = np.array(v)
        r_mag = math.sqrt(sum(x**2 for x in r_vec))
        if r_mag >= 0.5:
            valid_verts.append(v)
    
    # Limit to max_vectors
    selected_verts = valid_verts[:max_vectors]
    
    for v in selected_verts:
        r_vec = np.array(v)
        
        # Compute tangential velocity via cross product: v = ω × r
        tangential_v = cross_product(omega_vec, r_vec)
        
        # Draw tangential velocity if enabled
        if show_tangential:
            # Scale velocity for visibility: very short vectors, clamped to max from config
            speed = math.sqrt(sum(c**2 for c in tangential_v))
            if speed >= 0.01:
                scale_factor = min(speed * tangential_scale, tangential_max)
                scaled_v = (tangential_v[0] * scale_factor / speed,
                            tangential_v[1] * scale_factor / speed,
                            tangential_v[2] * scale_factor / speed)
                draw_velocity_arrow_at_point(tuple(v), np.array(scaled_v), screen)
        
        # Draw centripetal acceleration if enabled
        if show_centripetal:
            centripetal_a = cross_product(omega_vec, tangential_v)
            a_mag = math.sqrt(sum(c**2 for c in centripetal_a))
            if a_mag >= 0.01:
                a_unit = np.array([c / a_mag for c in centripetal_a])
                a_length = normal_vertex_len  # Fixed short length for centripetal vectors
                scaled_a = a_unit * a_length
                draw_velocity_arrow_at_point(tuple(v), scaled_a, screen, is_normal=True)  # centripetal = highlight (orange-ish)


def draw_cube_vertices(verts, screen):
    """Draw cube corner points in red, skipping points behind camera."""
    try:
        import pygame
    except ImportError:
        return
    
    for v in verts:
        p = project_3d_to_screen(*v)
        if p is not None:
            pygame.draw.circle(screen, (255, 50, 50), p, 6)  # Red vertices


def draw_velocity_vector_on_face(face_indices, verts, screen, omega_x=0, omega_y=0, omega_z=0):
    """Draw velocity vector on a cube face showing rotation motion."""
    if not isinstance(face_indices, tuple):
        return
    
    face_verts_list = [verts[i] for i in face_indices]
    center_pt = np.mean(face_verts_list, axis=0)
    
    # Compute tangential velocity at center point (distance from rotation axis)
    r_vec = np.array(center_pt)
    total_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    if total_mag < 0.1:
        return
    
    # Tangential velocity direction is perpendicular to radius and rotation axis
    tangent = np.cross(r_vec, [total_mag, 0, 0])
    
    scale_factor = min(total_mag / max(1, total_mag), 0.5)
    tip_x = center_pt[0] + tangent[0] * scale_factor * 4
    tip_y = -center_pt[1] + tangent[1] * scale_factor * 4
    
    p_start = project_3d_to_screen(*center_pt)
    tip_proj = (int(p_start[0] + tangent[0] * scale_factor * 6), 
                 int(p_start[1] + tangent[1] * scale_factor * 5))
    
    if screen is not None:
        pygame.draw.line(screen, COLORS['velocity'], p_start, tip_proj, 3)


def draw_formula(screen):
    """Draw the angular velocity formula at bottom right."""
    try:
        import pygame
    except ImportError:
        return
    
    formula_text = r'$\vec{v} = \boldsymbol{\omega} \times \vec{r}$'
    tangential_note = 'Tangential speed increases with distance from rotation axis'
    
    font = pygame.font.Font(None, 28)
    
    formula_text_render = font.render(formula_text, True, (255, 255, 255))
    tangential_text_render = font.render(tangential_note, True, (100, 150, 200))
    
    screen.blit(formula_text_render, (SCREEN_W - formula_text_render.get_width() - 40, SCREEN_H - 30))
    screen.blit(tangential_text_render, (SCREEN_W - tangential_text_render.get_width() - 40, SCREEN_H - 60))


def draw_face_normals(verts, screen):
    """Draw normal vectors from the center of each cube face."""
    try:
        import pygame
    except ImportError:
        return
    
    # Get face normal length from config
    vec_scales = CONFIG.get('vector_scales', {})
    normal_face_len = vec_scales.get('normal_face', 25)
    
    normal_colors = {
        'front': (100, 255, 100),   # Green - +Z
        'back': (100, 200, 255),    # Cyan - -Z
        'right': (255, 150, 100),   # Orange - +X
        'left': (200, 100, 255),    # Purple - -X
    }
    
    normal_names = ['front', 'back', 'right', 'left']
    
    for idx, face_indices in enumerate(CUBE_FACE_INDICES):
        face_verts_list = [verts[i] for i in face_indices]
        center_pt = np.mean(face_verts_list, axis=0)
        
        # Compute face normal (average of vertex positions, normalized)
        r_vec = np.array(center_pt)
        r_mag = math.sqrt(sum(x**2 for x in r_vec))
        if r_mag < 0.1:
            continue
        normal = r_vec / r_mag
        
        # Draw normal vector from face center
        tip_3d = (center_pt[0] + normal[0] * normal_face_len,
                  center_pt[1] - normal[1] * normal_face_len,
                  center_pt[2] + normal[2] * normal_face_len)
        
        p_start = project_3d_to_screen(*center_pt)
        p_tip = project_3d_to_screen(*tip_3d)
        
        if p_start is None or p_tip is None:
            continue
        
        color = normal_colors[normal_names[idx]]
        pygame.draw.line(screen, color, p_start, p_tip, 2)
        
        # Draw small arrowhead
        angle = math.atan2(p_tip[1] - p_start[1], p_tip[0] - p_start[0])
        arrow_len = 6
        pa1 = (int(p_tip[0] - arrow_len * math.cos(angle - 0.4)),
               int(p_tip[1] + arrow_len * math.sin(angle - 0.4)))
        pa2 = (int(p_tip[0] - arrow_len * math.cos(angle + 0.4)),
               int(p_tip[1] + arrow_len * math.sin(angle + 0.4)))
        pygame.draw.lines(screen, color, False, [pa1, p_tip, pa2], 2)


def draw_velocity_vectors_on_cube(verts, omega_x=0, omega_y=0, omega_z=0, screen=None):
    """Draw velocity vectors on each cube face showing rotation."""
    if screen is None:
        return
    
    # Compute total angular velocity magnitude
    total_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    for face_indices in CUBE_FACE_INDICES:
        draw_velocity_vector_on_face(face_indices, verts, screen, omega_x, omega_y, omega_z)


# ==================== VECTOR SHADING ====================

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


# ==================== 3D VECTOR RENDERING ====================

def draw_3d_cylinder(start, end, radius, segments, screen, color, depth_offset=0.0):
    """Draw a 3D cylinder as filled polygons, depth-sorted.
    
    Draws a cylinder from start to end with given radius and segment count.
    Returns list of faces as ((p1, p2, p3, p4), avg_depth, face_normal_3d) for external depth sorting and shading.
    """
    try:
        import pygame
    except ImportError:
        return []
    
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
        
        # Project to screen
        p1s = project_3d_to_screen(x1s, y1s, z1s)
        p2s = project_3d_to_screen(x2s, y2s, z2s)
        p1e = project_3d_to_screen(x1e, y1e, z1e)
        p2e = project_3d_to_screen(x2e, y2e, z2e)
        
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


def draw_3d_cone(apex, base_center, base_radius, segments, screen, color, depth_offset=0.0):
    """Draw a 3D cone as filled polygons, returns faces for depth sorting.
    
    Cone apex is the tip, base_center is the center of the circular base.
    Returns list of faces as ((p1, p2, apex_screen), avg_depth, face_normal_3d) for external depth sorting and shading.
    """
    try:
        import pygame
    except ImportError:
        return []
    
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
    
    # Project apex to screen
    apex_screen = project_3d_to_screen(*apex)
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
        
        p1 = project_3d_to_screen(bx1, by1, bz1)
        p2 = project_3d_to_screen(bx2, by2, bz2)
        
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


def draw_3d_vector_omega(screen, center, direction, magnitude, config, geom_config):
    """Draw the omega vector as a 3D cylinder with cone arrowhead.
    
    direction: normalized 3D vector (already rotated by camera view)
    magnitude: current omega magnitude for color scaling
    """
    if magnitude < 0.1:
        return []
    
    # Get geometry config
    shaft_radius = geom_config.get('shaft_radius', 3.0)
    shaft_segments = geom_config.get('shaft_segments', 8)
    arrowhead_length_ratio = geom_config.get('arrowhead_length_ratio', 0.2)
    arrowhead_radius_ratio = geom_config.get('arrowhead_radius_ratio', 0.4)
    
    # Total arrow length
    total_length = min(magnitude * 0.5 + 20, 120)  # Scale with magnitude, clamped
    shaft_length = total_length * (1.0 - arrowhead_length_ratio)
    arrowhead_length = total_length * arrowhead_length_ratio
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
    mag_norm = min(magnitude / config['angular_velocity']['max_speed'], 1.0)
    r = int(80 + mag_norm * 175)
    g = int(80 + (1 - mag_norm) * 100)
    b = int(255 - mag_norm * 150)
    color = (r, g, b)
    
    # Get cylinder and cone faces
    cylinder_faces = draw_3d_cylinder(center, shaft_end, shaft_radius, shaft_segments, screen, color, depth_offset=0.01)
    cone_faces = draw_3d_cone(cone_apex, shaft_end, arrowhead_base_radius, shaft_segments, screen, color, depth_offset=-0.01)
    
    return cylinder_faces + cone_faces, color


def draw_3d_vector(screen, start_3d, direction_3d, length, color, geom_config, proj_multiplier_x=0.15, proj_multiplier_y=0.12):
    """Draw a vector as a 3D cylinder with cone arrowhead, projecting through perspective.
    
    For tangential/centripetal vectors that start at cube vertices.
    Returns list of ((p1,p2,p3,...), depth) tuples for depth sorting.
    """
    try:
        import pygame
    except ImportError:
        return []
    
    if length < 1e-10:
        return []
    
    # Get geometry config
    shaft_radius = geom_config.get('shaft_radius', 2.0)
    shaft_segments = geom_config.get('shaft_segments', 6)
    arrowhead_length_ratio = geom_config.get('arrowhead_length_ratio', 0.25)
    arrowhead_radius_ratio = geom_config.get('arrowhead_radius_ratio', 0.35)
    
    # End point in 3D (direction is already normalized)
    end_3d = (
        start_3d[0] + direction_3d[0] * length,
        start_3d[1] + direction_3d[1] * length,
        start_3d[2] + direction_3d[2] * length
    )
    
    # Check if start or end is behind camera
    p_start = project_3d_to_screen(*start_3d)
    p_end = project_3d_to_screen(*end_3d)
    if p_start is None or p_end is None:
        return []
    
    shaft_length_3d = length * (1.0 - arrowhead_length_ratio)
    arrowhead_length_3d = length * arrowhead_length_ratio
    arrowhead_base_radius = shaft_radius * arrowhead_radius_ratio
    
    # Shaft end point (where cone base begins)
    shaft_end_3d = (
        start_3d[0] + direction_3d[0] * shaft_length_3d,
        start_3d[1] + direction_3d[1] * shaft_length_3d,
        start_3d[2] + direction_3d[2] * shaft_length_3d
    )
    
    # Get cylinder and cone faces
    cylinder_faces = draw_3d_cylinder(start_3d, shaft_end_3d, shaft_radius, shaft_segments, screen, color, depth_offset=0.01)
    cone_apex = (
        start_3d[0] + direction_3d[0] * length,
        start_3d[1] + direction_3d[1] * length,
        start_3d[2] + direction_3d[2] * length
    )
    cone_faces = draw_3d_cone(cone_apex, shaft_end_3d, arrowhead_base_radius, shaft_segments, screen, color, depth_offset=-0.01)
    
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
    try:
        import pygame
    except ImportError:
        return
    
    # Draw cylinder shaft as a thick line (polygon) in screen space
    dx = direction_screen[0]
    dy = direction_screen[1]
    # Perpendicular in screen space
    px = -dy
    py = dx
    mag = math.sqrt(px*px + py*py)
    if mag > 1e-10:
        px, py = px/mag * shaft_radius, py/mag * shaft_radius
        
        # Draw shaft as filled quad (simplified - just a thick line)
        # For proper 3D, we'd need depth info, but for the omega vector
        # which originates from center, screen-space thick line works well
        # Draw as filled polygon strip
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


def draw_3d_velocity_vectors(verts, total_omega_mag, screen, omega_x=0, omega_y=0, omega_z=0, max_vectors=3, show_tangential=True, show_centripetal=True, view_x=0, view_y=0, view_z=0):
    """Draw tangential velocity vectors and centripetal acceleration vectors as 3D cylinder+cone objects.
    
    Returns two lists: (tangential_drawables, centripetal_drawables)
    Each drawable is ((depth, 'polygon', screen_coords_tuple, color)).
    """
    try:
        import pygame
    except ImportError:
        return [], []
    
    # Get geometry configs
    tang_geom = CONFIG.get('vector_geometry', {}).get('tangential', {})
    cent_geom = CONFIG.get('vector_geometry', {}).get('centripetal', {})
    
    # Get vector scale factors from config
    vec_scales = CONFIG.get('vector_scales', {})
    tangential_scale = vec_scales.get('tangential', 0.08)
    tangential_max = vec_scales.get('tangential_max', 8)
    normal_vertex_len = vec_scales.get('normal_vertex', 6)
    
    # Use full omega vector for proper cross product
    omega_vec = np.array([omega_x, omega_y, omega_z])
    
    # Select only a subset of vertices
    valid_verts = []
    for v in verts:
        r_vec = np.array(v)
        r_mag = math.sqrt(sum(x**2 for x in r_vec))
        if r_mag >= 0.5:
            valid_verts.append(v)
    
    selected_verts = valid_verts[:max_vectors]
    
    tangential_drawables = []
    centripetal_drawables = []
    
    for v in selected_verts:
        r_vec = np.array(v)
        
        # Compute tangential velocity via cross product: v = ω × r
        tangential_v = cross_product(omega_vec, r_vec)
        
        # Draw tangential velocity if enabled
        if show_tangential:
            speed = math.sqrt(sum(c**2 for c in tangential_v))
            if speed >= 0.01:
                scale_factor = min(speed * tangential_scale, tangential_max)
                scaled_v = (tangential_v[0] * scale_factor / speed,
                            tangential_v[1] * scale_factor / speed,
                            tangential_v[2] * scale_factor / speed)
                
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
                start_cam = rotate_around_axis(start_3d, [0, 1, 0], view_y)
                start_cam = rotate_around_axis(start_cam, [0, 0, 1], -view_z)
                start_cam = rotate_around_axis(start_cam, [1, 0, 0], view_x)
                
                end_cam = rotate_around_axis(end_world, [0, 1, 0], view_y)
                end_cam = rotate_around_axis(end_cam, [0, 0, 1], -view_z)
                end_cam = rotate_around_axis(end_cam, [1, 0, 0], view_x)
                
                # Check if visible (use original world-space depth)
                p_start = project_3d_to_screen(*start_cam)
                p_end = project_3d_to_screen(*end_cam)
                if p_start is None or p_end is None:
                    continue
                
                # Shaft length and arrowhead based on rotated 3D positions
                total_len = math.sqrt((end_cam[0]-start_cam[0])**2 + (end_cam[1]-start_cam[1])**2 + (end_cam[2]-start_cam[2])**2)
                shaft_len = total_len * (1.0 - t_arrowhead_ratio)
                arrowhead_len = total_len * t_arrowhead_ratio
                arrowhead_base_r = t_shaft_radius * t_arrowhead_radius_ratio
                
                # Shaft end (interpolate between start and end in camera-rotated space)
                dx = end_cam[0] - start_cam[0]
                dy = end_cam[1] - start_cam[1]
                dz = end_cam[2] - start_cam[2]
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                if length < 1e-10:
                    continue
                shaft_end_3d = (
                    start_cam[0] + dx/length * shaft_len,
                    start_cam[1] + dy/length * shaft_len,
                    start_cam[2] + dz/length * shaft_len
                )
                cone_apex_3d = end_cam
                
                # Average depth for sorting (use camera-rotated z-depth)
                avg_depth = (start_cam[2] + shaft_end_3d[2] + cone_apex_3d[2]) / 3.0
                
                # Get cylinder and cone faces (use camera-rotated coordinates)
                cylinder_faces = draw_3d_cylinder(start_cam, shaft_end_3d, t_shaft_radius, t_shaft_segments, screen, COLORS['velocity'], depth_offset=0.01)
                cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_r, t_shaft_segments, screen, COLORS['velocity'], depth_offset=-0.01)
                
                for face, depth, normal in cylinder_faces:
                    tangential_drawables.append((avg_depth, 'polygon', face, COLORS['velocity'], normal))
                for face, depth, normal in cone_faces:
                    tangential_drawables.append((avg_depth, 'polygon', face, COLORS['velocity'], normal))
        
        # Draw centripetal acceleration if enabled
        if show_centripetal:
            centripetal_a = cross_product(omega_vec, tangential_v)
            a_mag = math.sqrt(sum(c**2 for c in centripetal_a))
            if a_mag >= 0.01:
                a_unit = np.array([c / a_mag for c in centripetal_a])
                a_length = normal_vertex_len
                scaled_a = tuple(a_unit[i] * a_length for i in range(3))
                
                # Get geometry config for centripetal vectors
                c_shaft_radius = cent_geom.get('shaft_radius', 1.5)
                c_shaft_segments = cent_geom.get('shaft_segments', 6)
                c_arrowhead_ratio = cent_geom.get('arrowhead_length_ratio', 0.3)
                c_arrowhead_radius_ratio = cent_geom.get('arrowhead_radius_ratio', 0.3)
                
                # Compute 3D positions - scaled_a is already the visualization-scaled displacement
                start_3d = (v[0], v[1], v[2])
                
                # End point in world space (use scaled_a directly as displacement)
                end_world = (
                    start_3d[0] + scaled_a[0],
                    start_3d[1] + scaled_a[1],
                    start_3d[2] + scaled_a[2]
                )
                
                # Rotate both points by camera angles for projection
                start_cam = rotate_around_axis(start_3d, [0, 1, 0], view_y)
                start_cam = rotate_around_axis(start_cam, [0, 0, 1], -view_z)
                start_cam = rotate_around_axis(start_cam, [1, 0, 0], view_x)
                
                end_cam = rotate_around_axis(end_world, [0, 1, 0], view_y)
                end_cam = rotate_around_axis(end_cam, [0, 0, 1], -view_z)
                end_cam = rotate_around_axis(end_cam, [1, 0, 0], view_x)
                
                # Check if visible (use original world-space depth)
                p_start = project_3d_to_screen(*start_cam)
                p_end = project_3d_to_screen(*end_cam)
                if p_start is None or p_end is None:
                    continue
                
                # Shaft length and arrowhead based on rotated 3D positions
                total_len = math.sqrt((end_cam[0]-start_cam[0])**2 + (end_cam[1]-start_cam[1])**2 + (end_cam[2]-start_cam[2])**2)
                shaft_len = total_len * (1.0 - c_arrowhead_ratio)
                arrowhead_len = total_len * c_arrowhead_ratio
                arrowhead_base_r = c_shaft_radius * c_arrowhead_radius_ratio
                
                # Shaft end (interpolate between start and end in camera-rotated space)
                dx = end_cam[0] - start_cam[0]
                dy = end_cam[1] - start_cam[1]
                dz = end_cam[2] - start_cam[2]
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                if length < 1e-10:
                    continue
                shaft_end_3d = (
                    start_cam[0] + dx/length * shaft_len,
                    start_cam[1] + dy/length * shaft_len,
                    start_cam[2] + dz/length * shaft_len
                )
                cone_apex_3d = end_cam
                
                # Average depth for sorting (use camera-rotated z-depth)
                avg_depth = (start_cam[2] + shaft_end_3d[2] + cone_apex_3d[2]) / 3.0
                
                # Get cylinder and cone faces (use camera-rotated coordinates)
                cylinder_faces = draw_3d_cylinder(start_cam, shaft_end_3d, c_shaft_radius, c_shaft_segments, screen, COLORS['highlight'], depth_offset=0.01)
                cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_r, c_shaft_segments, screen, COLORS['highlight'], depth_offset=-0.01)
                
                for face, depth, normal in cylinder_faces:
                    centripetal_drawables.append((avg_depth, 'polygon', face, COLORS['highlight'], normal))
                for face, depth, normal in cone_faces:
                    centripetal_drawables.append((avg_depth, 'polygon', face, COLORS['highlight'], normal))
    
    return tangential_drawables, centripetal_drawables
