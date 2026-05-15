"""
Angular Velocity Demo - Driver Module
"""

import pygame
import math
import os
import sys
import numpy as np

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from logic import (
    CONFIG, COLORS, SCREEN_W, SCREEN_H, CUBE_SIZE,
    FRAME_RATE, CUBE_VERTS, CUBE_EDGES, rotate_x, rotate_y, rotate_z,
    apply_3axis_rotation_matrix, apply_drag_impulse, apply_drag_impulse_damping,
    clamp_angular_velocity, project_3d_to_screen, draw_cube_edges,
    draw_cube_vertices, draw_velocity_vectors_at_vertices,
    draw_formula, draw_face_normals, rotate_around_axis,
    draw_3d_cylinder, draw_3d_cone, draw_3d_velocity_vectors,
    apply_vector_shading_to_face,
)

def run_simulation(duration_seconds=None, event_injector=None, screen_callback=None):
    """
    Runs the physics simulation loop.
    :param duration_seconds: If provided, stops simulation after this many seconds.
    :param event_injector: A callable that returns a list of pygame events to inject.
    :param screen_callback: A callable that receives the screen surface.
    :return: (omega_x, omega_y, omega_z) final angular velocity state.
    """
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption('Angular Velocity Demo: v = ω × r')
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    
    # 3-Axis angular velocity state (inertia)
    omega_x = 0.0
    omega_y = 0.0
    omega_z = 0.0
    
    # Accumulated angular position
    theta_x_deg = 0.0
    theta_y_deg = 0.0
    theta_z_deg = 0.0
    
    # View rotation angles (camera)
    view_x_deg = CONFIG['rotation_angles']['x']
    view_y_deg = CONFIG['rotation_angles']['y']
    view_z_deg = CONFIG['rotation_angles']['z']
    
    synthetic_test_mode = False
    running = True
    start_time = pygame.time.get_ticks()
    last_mouse_pos = None
    is_dragging = False
    
    # WASDEQ rotation sensitivity config
    keys_omega_sens = CONFIG.get('keys_omega_sensitivity', {})
    KEY_OMEGA_X = keys_omega_sens.get('omega_x', 40.0)   # W/S → X-axis rotation
    KEY_OMEGA_Y = keys_omega_sens.get('omega_y', 40.0)   # A/D → Y-axis rotation
    KEY_OMEGA_Z = keys_omega_sens.get('omega_z', 30.0)   # E/Q → Z-axis rotation
    
    # Vector display toggles (all start OFF)
    show_omega_vector = False
    show_tangential_vectors = False
    show_centripetal_vectors = False
    show_face_normals = False

    while running:
        # Check duration
        if duration_seconds and (pygame.time.get_ticks() - start_time) > (duration_seconds * 1000):
            running = False

        # Inject events if injector is provided
        if event_injector:
            for event in event_injector():
                pygame.event.post(event)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    print(f"DEBUG - ω: ({omega_x:.4f}, {omega_y:.4f}, {omega_z:.4f})")
                elif event.key == pygame.K_t:
                    synthetic_test_mode = not synthetic_test_mode
                elif event.key == pygame.K_o:  # Toggle omega vector
                    show_omega_vector = not show_omega_vector
                elif event.key == pygame.K_v:  # Toggle tangential velocity vectors
                    show_tangential_vectors = not show_tangential_vectors
                elif event.key == pygame.K_c:  # Toggle centripetal acceleration vectors
                    show_centripetal_vectors = not show_centripetal_vectors
                elif event.key == pygame.K_n:  # Toggle face normals
                    show_face_normals = not show_face_normals
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Stop the cube when LMB is first pressed (for aiming)
                omega_x, omega_y, omega_z = 0.0, 0.0, 0.0
                is_dragging = True
                last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEMOTION:
                # Apply drag impulse only during active drag (mouse moved while button down)
                # Map: vertical drag → omega_y (Y-axis rotation), horizontal drag → omega_x (X-axis rotation)
                if is_dragging and last_mouse_pos is not None:
                    motion_dx = event.pos[0] - last_mouse_pos[0]
                    motion_dy = -(event.pos[1] - last_mouse_pos[1])
                    mouse_sens = CONFIG.get('mouse_drag_sensitivity', {})
                    imp_x = motion_dy * mouse_sens.get('omega_x', 0.5)   # vertical → X-axis (flip)
                    imp_y = motion_dx * mouse_sens.get('omega_y', 0.4)   # horizontal → Y-axis
                    omega_x, omega_y, omega_z = apply_drag_impulse(omega_x, omega_y, omega_z, imp_x, imp_y)
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # Stop dragging when LMB is released
                is_dragging = False
            elif event.type == pygame.MOUSEWHEEL:
                delta = -event.y
                view_x_deg = max(-85, min(85, view_x_deg + delta * 0.2))
                view_y_deg = max(-85, min(85, view_y_deg + delta * 0.15))

        # WASDEQ continuous key input (flight-simulator style)
        frame_dt = 1.0 / FRAME_RATE
        keys = pygame.key.get_pressed()
        key_omega_x = 0.0
        key_omega_y = 0.0
        key_omega_z = 0.0
        if keys[pygame.K_w]:
            key_omega_x -= KEY_OMEGA_X
        if keys[pygame.K_s]:
            key_omega_x += KEY_OMEGA_X
        if keys[pygame.K_a]:
            key_omega_y += KEY_OMEGA_Y
        if keys[pygame.K_d]:
            key_omega_y -= KEY_OMEGA_Y
        if keys[pygame.K_e]:
            key_omega_z += KEY_OMEGA_Z
        if keys[pygame.K_q]:
            key_omega_z -= KEY_OMEGA_Z
        
        # Apply key-based angular acceleration (adds to existing omega each frame)
        if key_omega_x != 0 or key_omega_y != 0 or key_omega_z != 0:
            # Add acceleration each frame (like thrust in a flight simulator)
            if not (CONFIG.get('synthetic_test', {}).get('enabled', False) and synthetic_test_mode):
                omega_x += key_omega_x * frame_dt
                omega_y += key_omega_y * frame_dt
                omega_z += key_omega_z * frame_dt
        elif not (CONFIG.get('synthetic_test', {}).get('enabled', False) and synthetic_test_mode):
            omega_x, omega_y, omega_z = apply_drag_impulse_damping(omega_x, omega_y, omega_z)
        
        omega_x, omega_y, omega_z = clamp_angular_velocity(omega_x, omega_y, omega_z)
        theta_x_deg += math.degrees(omega_x * frame_dt)
        theta_y_deg += math.degrees(omega_y * frame_dt)
        theta_z_deg += math.degrees(omega_z * frame_dt)

        # Rendering
        screen.fill(COLORS['bg'])
        
        # Axis reference lines removed (they don't rotate with cube and can be confusing)
        
        # Get vector rendering mode settings
        vec_rendering = CONFIG.get('vector_rendering', {})
        omega_mode = vec_rendering.get('omega', 'enhanced')
        tangential_mode = vec_rendering.get('tangential', 'enhanced')
        centripetal_mode = vec_rendering.get('centripetal', 'simple')
        
        # Draw omega vector (angular velocity direction) from center
        total_omega = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        mag_norm = min(total_omega / CONFIG['angular_velocity']['max_speed'], 1.0) if CONFIG['angular_velocity']['max_speed'] > 0 else 0
        
        # Collect all drawable objects with depth for sorting: (depth, draw_func, args, color)
        drawables = []
        
        # Draw omega vector
        if show_omega_vector and total_omega > 0.1:
            if omega_mode == 'enhanced':
                # Use 3D cylinder+cone rendering
                vec_geom = CONFIG.get('vector_geometry', {}).get('omega', {})
                shaft_radius = vec_geom.get('shaft_radius', 3.0)
                shaft_segments = vec_geom.get('shaft_segments', 8)
                arrowhead_length_ratio = vec_geom.get('arrowhead_length_ratio', 0.2)
                arrowhead_radius_ratio = vec_geom.get('arrowhead_radius_ratio', 0.4)
                
                # Normalize omega vector and apply only camera view angles
                # The omega vector represents the axis of rotation in world space.
                # It should NOT rotate with the cube — it defines the fixed rotation axis.
                omega_unit = np.array([omega_x, omega_y, omega_z]) / total_omega
                # Apply camera rotation for view only (omega is in world/inertial frame)
                omega_rot = rotate_around_axis(omega_unit.tolist(), [0, 1, 0], view_y_deg)
                omega_rot = rotate_around_axis(omega_rot, [0, 0, 1], -view_z_deg)
                omega_rot = rotate_around_axis(omega_rot, [1, 0, 0], view_x_deg)
                
                # Length scales with speed
                omega_arrow_length = CONFIG.get('vector_scales', {}).get('omega_arrow', 60)
                total_length = omega_arrow_length * (0.5 + 0.5 * mag_norm)
                shaft_length = total_length * (1.0 - arrowhead_length_ratio)
                arrowhead_length = total_length * arrowhead_length_ratio
                arrowhead_base_radius = shaft_radius * arrowhead_radius_ratio
                
                # 3D positions
                center_3d = (0.0, 0.0, 0.0)
                shaft_end_3d = (
                    omega_rot[0] * shaft_length,
                    omega_rot[1] * shaft_length,
                    omega_rot[2] * shaft_length
                )
                cone_apex_3d = (
                    omega_rot[0] * total_length,
                    omega_rot[1] * total_length,
                    omega_rot[2] * total_length
                )
                
                # Color scales with speed
                r = int(80 + mag_norm * 175)
                g = int(80 + (1 - mag_norm) * 100)
                b = int(255 - mag_norm * 150)
                omega_color = (r, g, b)
                
                # Get cylinder faces with depth and normal
                cylinder_faces = draw_3d_cylinder(center_3d, shaft_end_3d, shaft_radius, shaft_segments, screen, omega_color, depth_offset=0.01)
                for face, depth, normal in cylinder_faces:
                    drawables.append((depth, 'polygon', face, omega_color, normal))
                
                # Get cone faces with depth and normal
                cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_radius, shaft_segments, screen, omega_color, depth_offset=-0.01)
                for face, depth, normal in cone_faces:
                    drawables.append((depth, 'polygon', face, omega_color, normal))
            else:
                # Use simple 2D line + arrowhead rendering
                # Normalize omega vector and apply only camera view angles
                # The omega vector represents the axis of rotation in world space.
                # It should NOT rotate with the cube — it defines the fixed rotation axis.
                omega_unit = np.array([omega_x, omega_y, omega_z]) / total_omega
                # Apply camera rotation for view only (omega is in world/inertial frame)
                omega_rot = rotate_around_axis(omega_unit.tolist(), [0, 1, 0], view_y_deg)
                omega_rot = rotate_around_axis(omega_rot, [0, 0, 1], -view_z_deg)
                omega_rot = rotate_around_axis(omega_rot, [1, 0, 0], view_x_deg)
                
                # Color scales with speed
                r = int(80 + mag_norm * 175)
                g = int(80 + (1 - mag_norm) * 100)
                b = int(255 - mag_norm * 150)
                omega_color = (r, g, b)
                
                # 3D positions
                center_3d = (0.0, 0.0, 0.0)
                omega_arrow_length = CONFIG.get('vector_scales', {}).get('omega_arrow', 60)
                total_length = omega_arrow_length * (0.5 + 0.5 * mag_norm)
                tip_3d = (
                    omega_rot[0] * total_length,
                    omega_rot[1] * total_length,
                    omega_rot[2] * total_length
                )
                
                # Project to screen
                p_center = project_3d_to_screen(*center_3d)
                p_tip = project_3d_to_screen(*tip_3d)
                
                if p_center is not None and p_tip is not None:
                    # Clamp to screen bounds
                    margin = 50
                    p_tip_clamped = (
                        max(margin, min(SCREEN_W - margin, p_tip[0])),
                        max(margin, min(SCREEN_H - margin, p_tip[1]))
                    )
                    # Draw simple line with arrowhead
                    pygame.draw.line(screen, omega_color, p_center, p_tip_clamped, 3)
                    
                    # Draw arrowhead
                    dx = p_tip_clamped[0] - p_center[0]
                    dy = p_tip_clamped[1] - p_center[1]
                    angle = math.atan2(dy, dx)
                    arrow_len = 12
                    perp_angle = angle + math.pi / 2
                    cone_radius = arrow_len * 0.5
                    base1 = (int(p_tip_clamped[0] + cone_radius * math.cos(perp_angle)),
                             int(p_tip_clamped[1] + cone_radius * math.sin(perp_angle)))
                    base2 = (int(p_tip_clamped[0] - cone_radius * math.cos(perp_angle)),
                             int(p_tip_clamped[1] - cone_radius * math.sin(perp_angle)))
                    pygame.draw.polygon(screen, omega_color, [p_tip_clamped, base1, base2])

        # Draw cube edges (with depth sorting)
        rotated_verts_list = []
        for point in CUBE_VERTS:
            rotated_point = apply_3axis_rotation_matrix(point, math.radians(theta_x_deg), math.radians(theta_y_deg), math.radians(theta_z_deg))
            rx = rotate_around_axis(rotated_point, [0, 1, 0], view_y_deg)
            rx = rotate_around_axis(rx, [0, 0, 1], -view_z_deg)
            ry = rotate_around_axis(rx, [1, 0, 0], view_x_deg)
            rotated_verts_list.append((ry[0], ry[1], ry[2]))
        rotated_verts = tuple(rotated_verts_list)
        
        # Get edge faces with depth for proper layering
        edge_faces = []
        for idx, (i, j) in enumerate(CUBE_EDGES):
            p1_screen = project_3d_to_screen(*(rotated_verts[i]))
            p2_screen = project_3d_to_screen(*(rotated_verts[j]))
            if p1_screen is not None and p2_screen is not None:
                avg_z = (rotated_verts[i][2] + rotated_verts[j][2]) / 2.0
                edge_faces.append((avg_z, p1_screen, p2_screen))
        edge_faces.sort(key=lambda e: e[0])  # back to front
        
        # Draw edges first (behind everything)
        for (avg_z, p1, p2) in edge_faces:
            pygame.draw.line(screen, (255, 255, 255), p1, p2, 3)
        
        # Compute world-space vertices (before camera rotation) for vector computation
        world_verts_list = []
        for point in CUBE_VERTS:
            rotated_point = apply_3axis_rotation_matrix(point, math.radians(theta_x_deg), math.radians(theta_y_deg), math.radians(theta_z_deg))
            world_verts_list.append((rotated_point[0], rotated_point[1], rotated_point[2]))
        world_verts = tuple(world_verts_list)
        
        # Collect vector drawables with depth
        vector_drawables = []
        
        # Draw tangential and centripetal vectors based on rendering mode
        if show_tangential_vectors or show_centripetal_vectors:
            if tangential_mode == 'enhanced' and centripetal_mode == 'enhanced':
                # Both enhanced - use 3D rendering with world-space vertices for correct cross product
                tangential_drawables, centripetal_drawables = draw_3d_velocity_vectors(
                    world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                    show_tangential=show_tangential_vectors, show_centripetal=show_centripetal_vectors,
                    view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg
                )
                vector_drawables.extend(tangential_drawables)
                vector_drawables.extend(centripetal_drawables)
            else:
                # Use simple 2D rendering for vectors
                if show_tangential_vectors and tangential_mode == 'simple':
                    draw_velocity_vectors_at_vertices(
                        world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                        max_vectors=3, show_tangential=True, show_centripetal=False,
                        view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg
                    )
                if show_centripetal_vectors and centripetal_mode == 'simple':
                    draw_velocity_vectors_at_vertices(
                        world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                        max_vectors=3, show_tangential=False, show_centripetal=True,
                        view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg
                    )
                # For mixed modes, use 3D but filter internally
                if show_tangential_vectors and tangential_mode == 'enhanced':
                    tang_drawables, _ = draw_3d_velocity_vectors(
                        world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                        show_tangential=True, show_centripetal=False,
                        view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg
                    )
                    vector_drawables.extend(tang_drawables)
                if show_centripetal_vectors and centripetal_mode == 'enhanced':
                    _, cent_drawables = draw_3d_velocity_vectors(
                        world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                        show_tangential=False, show_centripetal=True,
                        view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg
                    )
                    vector_drawables.extend(cent_drawables)
        
        # Combine all drawables and sort by depth (back to front)
        all_drawables = drawables + vector_drawables
        all_drawables.sort(key=lambda d: d[0])
        
        # Draw vertices (before vectors, after cube edges)
        draw_cube_vertices(rotated_verts, screen)
        
        # Draw all depth-sorted polygons with shading
        for drawable in all_drawables:
            depth = drawable[0]
            draw_type = drawable[1]
            args = drawable[2]
            color = drawable[3]
            face_normal = drawable[4] if len(drawable) > 4 else None
            if draw_type == 'polygon':
                if face_normal is not None:
                    shaded_color = apply_vector_shading_to_face(color, face_normal, depth)
                else:
                    shaded_color = color
                pygame.draw.polygon(screen, shaded_color, args, 0)  # filled
        
        # Draw face normals
        if show_face_normals:
            draw_face_normals(rotated_verts, screen)

        # UI Text - including toggle states
        toggle_info = font.render(f'[O] Omega: {"ON" if show_omega_vector else "OFF"}  [V] Tangential: {"ON" if show_tangential_vectors else "OFF"}  [C] Centripetal: {"ON" if show_centripetal_vectors else "OFF"}  [N] Normals: {"ON" if show_face_normals else "OFF"}  [WASDEQ] Rotate', True, (150, 150, 150))
        mag_text = f'Angular Velocity: |ω| = {total_omega:.2f} rad/s'
        info1 = font.render(mag_text, True, (255, 255, 255))
        screen.blit(info1, (15, 15))
        screen.blit(toggle_info, (15, 45))
        draw_formula(screen)
        
        pygame.display.flip()
        if screen_callback:
            screen_callback(pygame.display.get_surface())
        clock.tick(FRAME_RATE)

    pygame.quit()
    return omega_x, omega_y, omega_z

def main():
    run_simulation()

if __name__ == "__main__":
    main()