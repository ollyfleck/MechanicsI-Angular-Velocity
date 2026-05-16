"""
Math utilities module - Pure math functions for vectors and rotations.
No pygame or UI dependencies.
"""

import math
import numpy as np

from config import CONFIG, FRAME_RATE


# ==================== VECTOR OPERATIONS ====================

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


# ==================== ROTATION FUNCTIONS ====================

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


# ==================== PLANE PROJECTION ====================

def project_vector_to_plane(vec, normal):
    """Project a vector onto a plane with given normal."""
    dot = dot_product(vec, normalize_vector(normal))
    proj = vec - dot * normal
    return proj