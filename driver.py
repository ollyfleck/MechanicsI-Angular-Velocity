"""
Angular Velocity Demo - Driver Module
"""

import pygame
import math
import os
import sys

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Config and constants
from config import CONFIG, COLORS, SCREEN_W, SCREEN_H, CUBE_SIZE, FRAME_RATE

# Math utilities
from math_utils import (
    rotate_x, rotate_y, rotate_z, rotate_around_axis,
    apply_3axis_rotation_matrix,
)

# Projection
from projection import project_3d_to_screen

# Geometry
from geometry import CUBE_VERTS, CUBE_EDGES, CUBE_FACE_INDICES

# Physics
from physics import (
    apply_drag_impulse, apply_drag_impulse_damping,
    clamp_angular_velocity,
)

# Drawing
from draw_3d import (
    draw_3d_cylinder, draw_3d_cone, draw_3d_velocity_vectors,
    draw_3d_face_normals,
    apply_vector_shading_to_face,
    compute_vector_alpha,
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

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    pygame.display.set_caption('Angular Velocity Demo: v = ω × r')

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 24)

    # Fixed physics timestep for stable, frame-rate-independent integration
    fixed_physics_dt = CONFIG['physics'].get('fixed_physics_dt', 1.0 / 120.0)

    # Visual rotation settings - controls how fast the cube appears to spin visually
    visual_rotation_enabled = CONFIG.get('visual_rotation', {}).get('enabled', True)
    visual_multiplier = CONFIG.get('visual_rotation', {}).get('multiplier', 1.0)

    # Physics accumulator for fixed-timestep integration
    physics_accumulator = 0.0

    # 3-Axis angular velocity state (inertia)
    omega_x = 0.0
    omega_y = 0.0
    omega_z = 0.0

    # Accumulated orientation as quaternion (w, x, y, z) — starts at identity
    quat_w = 1.0
    quat_x = 0.0
    quat_y = 0.0
    quat_z = 0.0

    # View rotation angles (camera)
    view_x_deg = CONFIG['rotation_angles']['x']
    view_y_deg = CONFIG['rotation_angles']['y']
    view_z_deg = CONFIG['rotation_angles']['z']

    synthetic_test_mode = False
    running = True
    start_time = pygame.time.get_ticks()
    last_mouse_pos = None
    is_dragging = False

    # FPS tracking
    fps_frame_count = 0
    fps_check_time = pygame.time.get_ticks()
    display_fps = 0.0

    # WASDEQ rotation sensitivity config
    keys_omega_sens = CONFIG.get('keys_omega_sensitivity', {})
    KEY_OMEGA_X = keys_omega_sens.get('omega_x', 40.0)   # W/S → X-axis rotation
    KEY_OMEGA_Y = keys_omega_sens.get('omega_y', 40.0)   # A/D → Y-axis rotation
    KEY_OMEGA_Z = keys_omega_sens.get('omega_z', 30.0)   # E/Q → Z-axis rotation

    # Vector display toggle defaults - load from config
    display_toggles = CONFIG.get('display_toggles', {})
    show_omega_vector = display_toggles.get('show_omega_vector', False)
    show_tangential_vectors = display_toggles.get('show_tangential_vectors', False)
    show_centripetal_vectors = display_toggles.get('show_centripetal_vectors', False)
    show_face_normals = display_toggles.get('show_face_normals', False)

    # Per-vertex vector display toggles (keys 1-8) - load defaults from config
    vertex_config = CONFIG.get('vertices', {})
    vertex_vector_enabled = {}
    for i in range(8):
        vertex_key = f'vertex_{i}'
        if vertex_key in vertex_config:
            vertex_vector_enabled[i] = vertex_config[vertex_key].get('enabled', True)
        else:
            vertex_vector_enabled[i] = True  # Default to enabled if not configured

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
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.get_surface()
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
                elif event.key >= pygame.K_1 and event.key <= pygame.K_8:  # Toggle per-vertex vectors
                    vert_idx = event.key - pygame.K_1
                    vertex_vector_enabled[vert_idx] = not vertex_vector_enabled[vert_idx]
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Stop the cube when LMB is first pressed (for aiming)
                omega_x, omega_y, omega_z = 0.0, 0.0, 0.0
                is_dragging = True
                last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEMOTION:
                # Apply drag impulse only during active drag (mouse moved while button down)
                if is_dragging and last_mouse_pos is not None:
                    motion_dx = event.pos[0] - last_mouse_pos[0]
                    motion_dy = -(event.pos[1] - last_mouse_pos[1])
                    mouse_sens = CONFIG.get('mouse_drag_sensitivity', {})
                    imp_x = -motion_dy * mouse_sens.get('omega_x', 0.5)   # vertical → X-axis (negate)
                    imp_y = -motion_dx * mouse_sens.get('omega_y', 0.4)   # horizontal → Y-axis (negate)
                    omega_x, omega_y, omega_z = apply_drag_impulse(omega_x, omega_y, omega_z, imp_x, imp_y)
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                is_dragging = False
            elif event.type == pygame.MOUSEWHEEL:
                delta = -event.y
                view_x_deg = max(-85, min(85, view_x_deg + delta * 0.2))
                view_y_deg = max(-85, min(85, view_y_deg + delta * 0.15))

        # Get actual elapsed time since last frame (in seconds)
        frame_time_ms = clock.tick(FRAME_RATE)  # Caps FPS and returns ms elapsed
        real_dt = frame_time_ms / 1000.0  # Convert to seconds

        # FPS calculation - update every ~500ms
        fps_frame_count += 1
        current_time = pygame.time.get_ticks()
        if current_time - fps_check_time >= 500:  # Update every 500ms
            elapsed_s = (current_time - fps_check_time) / 1000.0
            if elapsed_s > 0:
                display_fps = int(fps_frame_count / elapsed_s)
            fps_frame_count = 0
            fps_check_time = current_time

        # WASDEQ continuous key input (flight-simulator style)
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

        # Fixed-timestep physics integration (frame-rate independent)
        if not (CONFIG.get('synthetic_test', {}).get('enabled', False) and synthetic_test_mode):
            # Accumulate real time and step physics at fixed intervals
            physics_accumulator += real_dt

            # Cap steps per frame to prevent spiral of death
            max_physics_steps = 256
            step_count = 0
            while physics_accumulator >= fixed_physics_dt and step_count < max_physics_steps:
                step_count += 1
                frame_dt = fixed_physics_dt  # Use fixed timestep for physics stability

                # Apply key-based angular acceleration (adds to existing omega each step)
                if key_omega_x != 0 or key_omega_y != 0 or key_omega_z != 0:
                    omega_x += key_omega_x * frame_dt
                    omega_y += key_omega_y * frame_dt
                    omega_z += key_omega_z * frame_dt

                # Apply damping at fixed timestep (compensate for increased damping frequency)
                # Skip damping when keyboard controls are actively being used
                keyboard_active = key_omega_x != 0 or key_omega_y != 0 or key_omega_z != 0
                if not keyboard_active:
                    effective_damping = apply_drag_impulse_damping(omega_x, omega_y, omega_z)
                    # Since we're applying damping more frequently at higher FPS,
                    # we need to reduce its per-step effect to maintain consistent behavior
                    current_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
                    if current_mag > 0:
                        max_speed = CONFIG['angular_velocity']['max_speed']
                        speed_factor = min(current_mag / max_speed, 1.0)
                        base_damping = CONFIG['angular_velocity']['damping']
                        adaptive_damping = base_damping - (speed_factor * 0.01)
                        # Scale damping to fixed timestep: if real_dt < fixed_physics_dt, less damping per step
                        damping_scale = min(real_dt / fixed_physics_dt, 1.0)
                        # Interpolate between no damping and full damping based on timestep ratio
                        step_damping = 1.0 - (1.0 - adaptive_damping) * damping_scale
                        omega_x *= step_damping
                        omega_y *= step_damping
                        omega_z *= step_damping

                physics_accumulator -= fixed_physics_dt

            # Clamp angular velocity after all physics steps
            omega_x, omega_y, omega_z = clamp_angular_velocity(omega_x, omega_y, omega_z)

        # Compute visual rotation scale based on current omega magnitude
        total_omega_current = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        max_speed = CONFIG['angular_velocity']['max_speed'] if CONFIG['angular_velocity']['max_speed'] > 0 else 180.0

        # Visual scale factor: apply user's direct multiplier
        if visual_rotation_enabled:
            visual_scale = visual_multiplier
        else:
            visual_scale = 1.0

        # Update accumulated orientation using quaternion integration with VISUAL scale.
        # Physics omega values remain unchanged for correct vector calculations,
        # but the visual rotation uses the scaled-down delta angle.
        if total_omega_current > 1e-10:
            # Use real_dt for continuous time-based integration (not fixed_physics_dt)
            # Apply visual_scale to reduce spin speed at high omega
            delta_angle_rad = total_omega_current * real_dt * visual_scale
            half_angle = delta_angle_rad / 2

            # Delta quaternion (fixed-frame rotation: q_delta ⊗ q_current)
            dq_w = math.cos(half_angle)
            dq_x = (omega_x / total_omega_current) * math.sin(half_angle)
            dq_y = (omega_y / total_omega_current) * math.sin(half_angle)
            dq_z = (omega_z / total_omega_current) * math.sin(half_angle)

            # q_new = q_delta ⊗ q_current (extrinsic/fixed-frame rotation order)
            new_w = dq_w*quat_w - dq_x*quat_x - dq_y*quat_y - dq_z*quat_z
            new_x = dq_w*quat_x + dq_x*quat_w + dq_y*quat_z - dq_z*quat_y
            new_y = dq_w*quat_y - dq_x*quat_z + dq_y*quat_w + dq_z*quat_x
            new_z = dq_w*quat_z + dq_x*quat_y - dq_y*quat_x + dq_z*quat_w

            # Normalize to prevent drift
            norm = math.sqrt(new_w**2 + new_x**2 + new_y**2 + new_z**2)
            if norm > 1e-20:
                quat_w = new_w / norm
                quat_x = new_x / norm
                quat_y = new_y / norm
                quat_z = new_z / norm

        # Rendering
        screen.fill(COLORS['bg'])
        
        # Create overlay surfaces for alpha-blended vectors
        # We need separate surfaces per alpha level to blend correctly
        overlay_surfaces = {}  # alpha -> pygame.Surface
        
        def get_overlay_surface(alpha):
            """Get or create an overlay surface for the given alpha level."""
            if alpha not in overlay_surfaces:
                curr_w, curr_h = screen.get_size()
                overlay = pygame.Surface((curr_w, curr_h), pygame.SRCALPHA)
                overlay.set_alpha(alpha)  # Set per-surface alpha for blending
                overlay_surfaces[alpha] = overlay
            return overlay_surfaces[alpha]

        # Get vector rendering mode settings from unified 'vectors' section
        vectors_cfg = CONFIG.get('vectors', {})
        omega_mode = vectors_cfg.get('omega', {}).get('rendering', 'enhanced')
        tangential_mode = vectors_cfg.get('tangential_velocity', {}).get('rendering', 'enhanced')
        centripetal_mode = vectors_cfg.get('centripetal_acceleration', {}).get('rendering', 'enhanced')

        total_omega = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        mag_norm = min(total_omega / max_speed, 1.0)

        # Collect all drawable objects with depth for sorting: (depth, draw_func, args, color)
        drawables = []

        # Draw omega vector - scale by real omega magnitude using scale_factor
        if show_omega_vector and total_omega > 0.01:
            # Get omega config from unified 'vectors' section
            omega_vec_cfg = vectors_cfg.get('omega', {})
            omega_scale = omega_vec_cfg.get('scale_factor', 0.5)
            
            # Get omega colors from config (fallback to defaults)
            omega_color = tuple(omega_vec_cfg.get('color', [255, 80, 80]))
            omega_color_secondary = tuple(omega_vec_cfg.get('color_secondary', [255, 255, 100]))
            omega_tip_length = omega_vec_cfg.get('geometry', {}).get('tip_length', 2.0)
            omega_clamp_max = omega_vec_cfg.get('clamp_max', 8)
            omega_min_length = omega_vec_cfg.get('min_length', 0)

            # Scale by omega magnitude, then clamp
            omega_display_length = total_omega * omega_scale
            omega_display_length = min(omega_display_length, omega_clamp_max)
            omega_display_length = max(omega_display_length, omega_min_length)

            # Get geometry config from unified 'vectors' section
            omega_geom = omega_vec_cfg.get('geometry', {})
            tip_length_omega = omega_geom.get('tip_length', 18.0)  # Fixed arrowhead length

            # Get geometry config for enhanced mode
            shaft_radius = omega_geom.get('shaft_radius', 3.0)
            shaft_segments = omega_geom.get('shaft_segments', 8)
            arrowhead_radius_ratio = omega_geom.get('arrowhead_radius_ratio', 5)

            # Draw omega vector with alpha support - remove tip_length check so it draws even when below tip
            if omega_mode == 'enhanced':
                omega_unit = omega_x / total_omega, omega_y / total_omega, omega_z / total_omega
                # Apply camera rotation for view only (omega is in world/inertial frame)
                omega_rot = rotate_around_axis(omega_unit, [0, 1, 0], view_y_deg)
                omega_rot = rotate_around_axis(omega_rot, [0, 0, 1], -view_z_deg)
                omega_rot = rotate_around_axis(omega_rot, [1, 0, 0], view_x_deg)

                # Blend between primary and secondary color based on display length ratio
                # Blending starts at tip_length (100% opacity) and ends at clamp_max
                if omega_clamp_max > omega_tip_length:
                    omega_blend = (omega_display_length - omega_tip_length) / (omega_clamp_max - omega_tip_length)
                else:
                    omega_blend = 0.0
                omega_blend = max(0.0, min(1.0, omega_blend))
                r = int(omega_color[0] * (1 - omega_blend) + omega_color_secondary[0] * omega_blend)
                g = int(omega_color[1] * (1 - omega_blend) + omega_color_secondary[1] * omega_blend)
                b = int(omega_color[2] * (1 - omega_blend) + omega_color_secondary[2] * omega_blend)
                omega_color_final = (min(255, r), min(255, g), min(255, b))

                # Use draw_3d_vector_omega_drawables for alpha support
                # Compute total_length for the drawables function
                total_length = omega_display_length
                center_3d = (0.0, 0.0, 0.0)
                
                # Get cylinder and cone faces with alpha
                alpha = compute_vector_alpha(total_length, tip_length_omega)
                
                shaft_length = max(total_length - tip_length_omega, 0.01)
                arrowhead_base_radius = shaft_radius * arrowhead_radius_ratio
                
                shaft_end_3d = (omega_rot[0] * shaft_length, omega_rot[1] * shaft_length, omega_rot[2] * shaft_length)
                cone_apex_3d = (omega_rot[0] * total_length, omega_rot[1] * total_length, omega_rot[2] * total_length)

                cylinder_faces = draw_3d_cylinder(center_3d, shaft_end_3d, shaft_radius, shaft_segments, screen, omega_color_final, depth_offset=0.01)
                for face, depth, normal in cylinder_faces:
                    drawables.append((depth, 'polygon', face, omega_color_final, normal, alpha))

                cone_faces = draw_3d_cone(cone_apex_3d, shaft_end_3d, arrowhead_base_radius, shaft_segments, screen, omega_color_final, depth_offset=-0.01)
                for face, depth, normal in cone_faces:
                    drawables.append((depth, 'polygon', face, omega_color_final, normal, alpha))
                
                # Draw a red circle at the base of the omega vector's shaft
                curr_w, curr_h = screen.get_size()
                p_center_circle = project_3d_to_screen(*center_3d, curr_w, curr_h)
                if p_center_circle is not None:
                    drawables.append((center_3d[2] + 0.1, 'circle', (p_center_circle, shaft_radius), (255, 0, 0)))
            # Also draw in simple mode even when below tip_length - alpha handles visibility
            elif omega_mode == 'simple':
                omega_unit = (omega_x / total_omega, omega_y / total_omega, omega_z / total_omega)
                omega_rot = rotate_around_axis(omega_unit, [0, 1, 0], view_y_deg)
                omega_rot = rotate_around_axis(omega_rot, [0, 0, 1], -view_z_deg)
                omega_rot = rotate_around_axis(omega_rot, [1, 0, 0], view_x_deg)

                # Blend between primary and secondary color based on display length ratio
                # Blending starts at tip_length (100% opacity) and ends at clamp_max
                if omega_clamp_max > omega_tip_length:
                    omega_blend = (omega_display_length - omega_tip_length) / (omega_clamp_max - omega_tip_length)
                else:
                    omega_blend = 0.0
                omega_blend = max(0.0, min(1.0, omega_blend))
                r = int(omega_color[0] * (1 - omega_blend) + omega_color_secondary[0] * omega_blend)
                g = int(omega_color[1] * (1 - omega_blend) + omega_color_secondary[1] * omega_blend)
                b = int(omega_color[2] * (1 - omega_blend) + omega_color_secondary[2] * omega_blend)
                omega_color_final = (min(255, r), min(255, g), min(255, b))

                center_3d = (0.0, 0.0, 0.0)
                total_length = omega_display_length
                tip_3d = (omega_rot[0] * total_length, omega_rot[1] * total_length, omega_rot[2] * total_length)

                curr_w, curr_h = screen.get_size()
                p_center = project_3d_to_screen(*center_3d, curr_w, curr_h)
                p_tip = project_3d_to_screen(*tip_3d, curr_w, curr_h)

                if p_center is not None and p_tip is not None:
                    margin = 50
                    curr_w, curr_h = screen.get_size()
                    p_tip_clamped = (max(margin, min(curr_w - margin, p_tip[0])), max(margin, min(curr_h - margin, p_tip[1])))
                    
                    # Draw a circle at the base of the omega vector's shaft
                    pygame.draw.circle(screen, (255, 0, 0), (int(p_center[0]), int(p_center[1])), int(shaft_radius))
                    
                    pygame.draw.line(screen, omega_color_final, p_center, p_tip_clamped, 3)

                    dx2 = p_tip_clamped[0] - p_center[0]
                    dy2 = p_tip_clamped[1] - p_center[1]
                    angle = math.atan2(dy2, dx2)
                    arrow_len = 12
                    perp_angle = angle + math.pi / 2
                    cone_radius = arrow_len * 0.5
                    base1 = (int(p_tip_clamped[0] + cone_radius * math.cos(perp_angle)), int(p_tip_clamped[1] + cone_radius * math.sin(perp_angle)))
                    base2 = (int(p_tip_clamped[0] - cone_radius * math.cos(perp_angle)), int(p_tip_clamped[1] - cone_radius * math.sin(perp_angle)))
                    pygame.draw.polygon(screen, omega_color_final, [p_tip_clamped, base1, base2])

        # Compute world-space vertices using quaternion-based rotation.
        # The quaternion accumulates fixed-frame rotations from omega, which is exactly the physical orientation.
        # We apply this as a single matrix multiply to avoid Euler-order ambiguity.
        qw, qx, qy, qz = quat_w, quat_x, quat_y, quat_z
        xx = qx*qx; yy = qy*qy; zz = qz*qz
        xy = qx*qy; xz = qx*qz; yz = qy*qz
        wx = qw*qx; wy = qw*qy; wz = qw*qz

        # Rotation matrix (row-major) for quaternion-to-matrix conversion
        m00 = 1 - 2*(yy + zz);   m01 = 2*(xy - wz);       m02 = 2*(xz + wy)
        m10 = 2*(xy + wz);        m11 = 1 - 2*(xx + zz);   m12 = 2*(yz - wx)
        m20 = 2*(xz - wy);        m21 = 2*(yz + wx);        m22 = 1 - 2*(xx + yy)

        rotated_verts_array = [[0.0, 0.0, 0.0] for _ in range(len(CUBE_VERTS))]
        for i in range(len(CUBE_VERTS)):
            x, y, z = CUBE_VERTS[i]
            rotated_verts_array[i][0] = m00*x + m01*y + m02*z
            rotated_verts_array[i][1] = m10*x + m11*y + m12*z
            rotated_verts_array[i][2] = m20*x + m21*y + m22*z

        rotated_verts = tuple(tuple(rotated_verts_array[i]) for i in range(len(CUBE_VERTS)))

        # Get edge faces with depth for proper layering
        edge_faces = []
        for idx, (i, j) in enumerate(CUBE_EDGES):
            curr_w, curr_h = screen.get_size()
            p1_screen = project_3d_to_screen(*(rotated_verts[i]), curr_w, curr_h)
            p2_screen = project_3d_to_screen(*(rotated_verts[j]), curr_w, curr_h)
            if p1_screen is not None and p2_screen is not None:
                avg_z = (rotated_verts[i][2] + rotated_verts[j][2]) / 2.0
                edge_faces.append((avg_z, p1_screen, p2_screen))
        edge_faces.sort(key=lambda e: e[0])  # back to front

        world_verts = rotated_verts  # quaternion-based rotation gives correct world-space vertices

        # Collect vector drawables with depth
        vector_drawables = []

        # Draw tangential and centripetal vectors based on rendering mode
        if show_tangential_vectors or show_centripetal_vectors:
            # Build vertex mask from per-vertex toggles
            vertex_mask = dict(vertex_vector_enabled)
            # When per-vertex toggles are used, show all enabled vertices (not limited to 3)
            max_display_vectors = 8
            if tangential_mode == 'enhanced' and centripetal_mode == 'enhanced':
                tangential_drawables, centripetal_drawables = draw_3d_velocity_vectors(
                    world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                    show_tangential=show_tangential_vectors, show_centripetal=show_centripetal_vectors,
                    view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg,
                    max_vectors=max_display_vectors,
                    vertex_mask=vertex_mask,
                    width=curr_w, height=curr_h
                )
                vector_drawables.extend(tangential_drawables)
                vector_drawables.extend(centripetal_drawables)
            else:
                # Simple mode removed with legacy 2D code - use enhanced mode
                if show_tangential_vectors:
                    tang_drawables, _ = draw_3d_velocity_vectors(
                        world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                        show_tangential=True, show_centripetal=False,
                        view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg,
                        vertex_mask=vertex_mask,
                        width=curr_w, height=curr_h
                    )
                    vector_drawables.extend(tang_drawables)
                if show_centripetal_vectors:
                    _, cent_drawables = draw_3d_velocity_vectors(
                        world_verts, total_omega, screen, omega_x, omega_y, omega_z,
                        show_tangential=False, show_centripetal=True,
                        view_x=view_x_deg, view_y=view_y_deg, view_z=view_z_deg,
                        vertex_mask=vertex_mask,
                        width=curr_w, height=curr_h
                    )
                    vector_drawables.extend(cent_drawables)

        # Separate omega vector drawables from velocity vector drawables
        # Omega vector is drawn behind the cube (as a rotation axis indicator)
        omega_drawables = [d for d in drawables]
        velocity_drawables = vector_drawables

        # Combine omega drawables, edge faces, and vertex data for unified depth sorting
        # Create edge drawable entries: (depth, 'line', (p1, p2), color)
        edge_drawables = []
        for (avg_z, p1, p2) in edge_faces:
            edge_drawables.append((avg_z, 'line', (p1, p2), COLORS['cube_edges']))

        # Create vertex drawable entries: (depth, 'vertex', screen_pos, color)
        # Only draw vertex dots for enabled vertices
        vertex_drawables = []
        for i, v in enumerate(rotated_verts):
            curr_w, curr_h = screen.get_size()
            p = project_3d_to_screen(*v, curr_w, curr_h)
            if p is not None and vertex_vector_enabled.get(i, False):
                vertex_drawables.append((v[2], 'vertex', p, (255, 50, 50)))

        # Combine all drawables and sort by depth (back to front)
        # In this coord system: higher Z = farther from camera, lower Z = closer to camera
        # So we sort reverse=True: highest Z (farthest) drawn first/bottom, lowest Z (closest) drawn last/top
        all_drawables = omega_drawables + edge_drawables + velocity_drawables + vertex_drawables
        
        # Add face normals to drawables if enabled (for depth sorting)
        if show_face_normals:
            normal_drawables = draw_3d_face_normals(rotated_verts, screen, width=curr_w, height=curr_h)
            all_drawables.extend(normal_drawables)
        
        all_drawables.sort(key=lambda d: d[0], reverse=True)

        # Draw all depth-sorted polygons, lines, and vertices with shading
        # Alpha-blended drawables go to overlay surfaces instead
        for drawable in all_drawables:
            depth = drawable[0]
            draw_type = drawable[1]
            args = drawable[2]
            color = drawable[3]
            face_normal = drawable[4] if len(drawable) > 4 else None
            alpha = drawable[5] if len(drawable) > 5 else None
            
            if draw_type == 'polygon':
                if face_normal is not None:
                    shaded_color = apply_vector_shading_to_face(color, face_normal, depth)
                else:
                    shaded_color = color
                
                if alpha is not None and alpha < 255:
                    # Draw to overlay surface for alpha blending
                    overlay = get_overlay_surface(alpha)
                    pygame.draw.polygon(overlay, shaded_color, args, 0)
                else:
                    pygame.draw.polygon(screen, shaded_color, args, 0)  # filled
            elif draw_type == 'line':
                p1, p2 = args
                pygame.draw.line(screen, color, p1, p2, 3)
            elif draw_type == 'vertex':
                pos = args
                pygame.draw.circle(screen, color, pos, 6)
            elif draw_type == 'circle':
                (cx, cy), radius = args
                pygame.draw.circle(screen, color, (int(cx), int(cy)), int(radius))
        
        # Blit all overlay surfaces onto the main screen
        for alpha, overlay in overlay_surfaces.items():
            screen.blit(overlay, (0, 0))


        # UI Text - including toggle states and FPS
        toggle_info = font.render(f'[O] Omega: {"ON" if show_omega_vector else "OFF"}  [V] Tangential: {"ON" if show_tangential_vectors else "OFF"}  [C] Centripetal: {"ON" if show_centripetal_vectors else "OFF"}  [N] Normals: {"ON" if show_face_normals else "OFF"}  [1-8] Vertex: {"ON" if any(vertex_vector_enabled.values()) else "OFF"}  [WASDEQ] Rotate', True, (150, 150, 150))
        omega_text = f'Angular Velocity: ω = ({omega_x:+.2f}, {omega_y:+.2f}, {omega_z:+.2f}) rad/s'
        info1 = font.render(omega_text, True, (255, 255, 255))

        # Visual scale indicator and FPS display
        fps_text = f'FPS: {display_fps}'
        fps_render = small_font.render(fps_text, True, (180, 180, 180))

        screen.blit(info1, (15, 15))
        screen.blit(toggle_info, (15, 45))
        if visual_rotation_enabled and abs(visual_multiplier - 1.0) > 0.01:
            scale_render = small_font.render(f'Visual Multiplier: {visual_multiplier:.2f}x', True, (200, 200, 100))
            screen.blit(scale_render, (15, 75))
        
        curr_w, curr_h = screen.get_size()
        screen.blit(fps_render, (curr_w - 100, 15))

        pygame.display.flip()
        if screen_callback:
            screen_callback(pygame.display.get_surface())

    pygame.quit()
    return omega_x, omega_y, omega_z

def main():
    run_simulation()

if __name__ == "__main__":
    main()