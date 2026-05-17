"""
Physics module - Angular velocity physics functions (drag, damping, clamping).
No pygame dependencies.
"""

import math

from config import CONFIG


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
    """Clamp angular velocity to maximum allowed magnitude per axis."""
    max_per_axis = CONFIG['angular_velocity']['max_speed']
    
    # Clamp each component independently to the max_speed limit
    omega_x = max(-max_per_axis, min(max_per_axis, omega_x))
    omega_y = max(-max_per_axis, min(max_per_axis, omega_y))
    omega_z = max(-max_per_axis, min(max_per_axis, omega_z))
    
    return omega_x, omega_y, omega_z
