"""
2D Drawing module - Simple 2D drawing functions for the cube visualization.
Requires pygame.
"""

import math
import numpy as np

try:
    import pygame
except ImportError:
    raise RuntimeError("pygame is required for drawing functions. Run: pip install pygame")

from config import COLORS, CONFIG, CUBE_SIZE, SCREEN_W, SCREEN_H, VECTOR_CONFIGS
from projection import project_3d_to_screen
from geometry import CUBE_EDGES, CUBE_FACE_INDICES, CUBE_VERTS
from math_utils import cross_product


# ==================== 2D DRAWING FUNCTIONS ====================

def draw_cube_edges(verts, screen):
    """Draw the cube wireframe edges in white, sorted back-to-front for depth."""
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


def draw_cube_vertices(verts, screen):
    """Draw cube corner points in red, skipping points behind camera."""
    for v in verts:
        p = project_3d_to_screen(*v)
        if p is not None:
            pygame.draw.circle(screen, (255, 50, 50), p, 6)  # Red vertices


def draw_velocity_arrow_at_point(point_3d, velocity_vec, screen, is_normal=False):
    """Draw a velocity or normal vector arrow at a point.
    
    For tangential vectors: uses right-hand rule v = omega x r
    For normal vectors: shows outward direction from face center.
    Arrow length scales with speed and arrowhead points in the vector direction.
    Uses vector_scales config for projection multipliers.
    """
    if len(velocity_vec) < 3:
        return

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
    - Tangential velocity (blue): v = omega x r, direction of motion due to rotation
    - Centripetal acceleration (orange): a = omega x v = omega x (omega x r), points toward rotation axis
    
    Arrow length scales linearly with magnitude from min_length to max_length.
    Uses vector_visualization config for scale factors.
    Toggled by show_tangential and show_centripetal flags.
    
    Note: Centripetal acceleration a = omega x v has magnitude |a| = |omega| * |v| = |omega|^2 * r,
    which scales quadratically with omega. We display it linearly from min to max based on
    normalized omega magnitude for consistent visual scaling.
    """
    # Get vector visualization config
    _, tang_cfg, cent_cfg, _ = VECTOR_CONFIGS
    
    max_speed = CONFIG['angular_velocity']['max_speed']
    speed_at_max_omega = max_speed * CUBE_SIZE          # reference for tangential velocity (linear in omega)
    ref_a_mag = max_speed * max_speed * CUBE_SIZE      # reference for centripetal acceleration (quadratic in omega)
    
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
        
        # Compute tangential velocity via cross product: v = omega x r
        tangential_v = cross_product(omega_vec, r_vec)
        
        # Draw tangential velocity if enabled
        if show_tangential:
            speed = math.sqrt(sum(c**2 for c in tangential_v))
            
            # Linear interpolation from min_length to max_length based on speed ratio
            speed_at_max_omega = max_speed * CUBE_SIZE  # reference speed when omega is at max
            speed_ratio = speed / speed_at_max_omega if speed_at_max_omega > 0 else 0
            
            min_len = tang_cfg.get('min_length', 0)
            max_len = tang_cfg.get('max_length', 8)
            
            # Linear scaling from min to max based on normalized speed
            clamped_ratio = min(max(speed_ratio, 0.0), 1.0)
            display_length = min_len + clamped_ratio * (max_len - min_len)
            
            if display_length > 0.01:
                # Create scaled vector in the direction of tangential_v
                if speed >= 0.01:
                    scaled_v = tuple(tangential_v[i] / speed * display_length for i in range(3))
                    draw_velocity_arrow_at_point(tuple(v), np.array(scaled_v), screen)
        
        # Draw centripetal acceleration if enabled
        if show_centripetal:
            centripetal_a = cross_product(omega_vec, tangential_v)
            a_mag = math.sqrt(sum(c**2 for c in centripetal_a))
            
            length_at_max = cent_cfg.get('length_at_max', 8)
            min_len = cent_cfg.get('min_length', 0)
            
            # Scale centripetal vector linearly from min to max based on normalized omega
            # Use quadratic reference (max_speed^2 * CUBE_SIZE) so it scales linearly with display
            cent_speed_ratio = a_mag / ref_a_mag if ref_a_mag > 0 else 0
            clamped_ratio = min(max(cent_speed_ratio, 0.0), 1.0)
            display_length = min_len + clamped_ratio * (length_at_max - min_len)
            
            if display_length > 0.01 and a_mag >= 0.01:
                a_unit = np.array([c / a_mag for c in centripetal_a])
                scaled_a = a_unit * display_length
                draw_velocity_arrow_at_point(tuple(v), scaled_a, screen, is_normal=True)


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
    
    pygame.draw.line(screen, COLORS['velocity'], p_start, tip_proj, 3)


def draw_velocity_vectors_on_cube(verts, omega_x=0, omega_y=0, omega_z=0, screen=None):
    """Draw velocity vectors on each cube face showing rotation."""
    if screen is None:
        return
    
    # Compute total angular velocity magnitude
    total_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    for face_indices in CUBE_FACE_INDICES:
        draw_velocity_vector_on_face(face_indices, verts, screen, omega_x, omega_y, omega_z)


def draw_formula(screen):
    """Draw the angular velocity formula at bottom right."""
    formula_text = r'$\vec{v} = \boldsymbol{\omega} \times \vec{r}$'
    tangential_note = 'Tangential speed increases with distance from rotation axis'
    
    font = pygame.font.Font(None, 28)
    
    formula_text_render = font.render(formula_text, True, (255, 255, 255))
    tangential_text_render = font.render(tangential_note, True, (100, 150, 200))
    
    screen.blit(formula_text_render, (SCREEN_W - formula_text_render.get_width() - 40, SCREEN_H - 30))
    screen.blit(tangential_text_render, (SCREEN_W - tangential_text_render.get_width() - 40, SCREEN_H - 60))


def draw_face_normals(verts, screen):
    """Draw normal vectors from the center of each cube face."""
    # Get face normal length from vector_visualization config
    _, _, _, face_cfg = VECTOR_CONFIGS
    normal_face_len = face_cfg.get('fixed_length', 3)
    
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
                  center_pt[1] + normal[1] * normal_face_len,
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