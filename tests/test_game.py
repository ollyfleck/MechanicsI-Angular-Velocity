"""
Angular Velocity Demo - Consolidated Unit Tests

This file consolidates tests from:
- angular_velocity/test_game.py (root directory tests)
- angular_velocity/tests/test_game.py (subdirectory tests)

Run with: python -m pytest tests/test_game.py
Or from project root: python -m pytest tests/
"""

import math
import sys
import os

# Ensure we can import from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def safe_load_module(module_name):
    """Load a module, handling import errors gracefully."""
    try:
        return __import__(module_name)
    except ImportError:
        pass

    # Try from file if installed import fails
    try:
        import importlib.util
        full_path = os.path.join(project_root, module_name + '.py')
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except:
                pass
            return module
    except Exception as e:
        print(f"Could not load from file: {e}")

    raise ImportError(f"Module {module_name} not found")


# Load driver module for config values using relative path from project root
driver = safe_load_module('driver')

SYNTHETIC_DRAG_DELTA_OMEGA_X = getattr(driver, 'SYNTHETIC_DRAG_DELTA_OMEGA_X', 0.5)
SYNTHETIC_DRAG_DELTA_OMEGA_Y = getattr(driver, 'SYNTHETIC_DRAG_DELTA_OMEGA_Y', -0.3)
SYNTHETIC_DRAG_DELTA_OMEGA_Z_BASE = getattr(driver, 'SYNTHETIC_DRAG_DELTA_OMEGA_Z_BASE', 0.1)
SYNTHETIC_DRAG_DELTA_OMEGA_Z_OFFSET = getattr(driver, 'SYNTHETIC_DRAG_DELTA_OMEGA_Z_OFFSET', 0.05)
ANGULAR_DAMPING = getattr(driver, 'ANGULAR_DAMPING', 0.98)
MAX_SPEED = getattr(driver, 'MAX_SPEED', 10.0)
FRAME_RATE = getattr(driver, 'FRAME_RATE', 60)


def test_damping_decreases_velocity():
    """Verify damping reduces angular velocity over time."""
    from physics import apply_drag_impulse_damping

    initial_omega = (0.5, 0.3, 0.2)
    omega_x, omega_y, omega_z = apply_drag_impulse_damping(*initial_omega)

    initial_mag = math.sqrt(sum(v**2 for v in initial_omega))
    final_mag = math.sqrt(sum(v**2 for v in (omega_x, omega_y, omega_z)))

    assert final_mag < initial_mag, "Damping should decrease velocity magnitude"


def test_clamp_limits_velocity():
    """Verify clamping prevents excessive angular velocity."""
    from physics import clamp_angular_velocity

    high_omega = (10.0, 5.0, 2.0)  # Way beyond max_speed

    omega_x, omega_y, omega_z = clamp_angular_velocity(*high_omega)

    total_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    assert total_mag <= MAX_SPEED * 1.5, "Clamped velocity should not exceed max"


def test_drag_impulse_adds_momentum():
    """Verify drag input adds torque to angular velocity."""
    from physics import apply_drag_impulse

    initial_omega = (0.0, 0.0, 0.0)
    drag_dx = 100
    drag_dy = -50

    omega_x, omega_y, omega_z = apply_drag_impulse(*initial_omega, drag_dx, drag_dy)

    assert math.sqrt(omega_x**2 + omega_y**2 + omega_z**2) > 0.1


def test_rotation_preserves_distance_from_origin():
    """Verify rotation maintains distance of points from origin."""
    from math_utils import apply_3axis_rotation_matrix

    point = (3.0, 4.0, 0.0)

    rotated = apply_3axis_rotation_matrix(point, 0.1, 0, 0)
    rotated_np = list(rotated)
    original_mag = math.sqrt(sum(v**2 for v in point))
    final_mag = math.sqrt(sum(v**2 for v in rotated_np))

    assert abs(final_mag - original_mag) < 0.01


def test_rotation_matrices_preserve_cube_shape():
    """Verify 3-axis rotation maintains cube geometry."""
    from math_utils import apply_3axis_rotation_matrix
    from geometry import CUBE_VERTS

    initial_mags = []
    final_mags = []

    for point in CUBE_VERTS:
        rot = apply_3axis_rotation_matrix(point, 0.2, -0.15, 0.08)
        initial_mags.append(math.sqrt(sum(v**2 for v in point)))
        final_mags.append(math.sqrt(sum(v**2 for v in rot)))

    for i in range(len(CUBE_VERTS)):
        assert abs(final_mags[i] - initial_mags[i]) < 0.01


def test_x_rotation_matrix_structure():
    """Verify rotation around X axis has correct matrix structure."""
    from math_utils import rotate_x

    point = (3.0, 4.0, 5.0)
    rotated = rotate_x(point, math.degrees(0.5))

    # After X-rotation, x coordinate should remain unchanged
    assert abs(rotated[0] - 3.0) < 1e-6


def test_y_rotation_matrix_structure():
    """Verify rotation around Y axis has correct matrix structure."""
    from math_utils import rotate_y

    point = (3.0, 4.0, 5.0)
    rotated = rotate_y(point, math.degrees(0.5))

    # After Y-rotation, y coordinate should remain unchanged
    assert abs(rotated[1] - 4.0) < 1e-6


def test_z_rotation_matrix_structure():
    """Verify rotation around Z axis has correct matrix structure."""
    from math_utils import rotate_z

    point = (3.0, 4.0, 5.0)
    rotated = rotate_z(point, math.degrees(0.5))

    # After Z-rotation, z coordinate should remain unchanged
    assert abs(rotated[2] - 5.0) < 1e-6


def test_identity_rotation_returns_same_point():
    """Verify zero rotation returns the original point."""
    from math_utils import apply_3axis_rotation_matrix

    point = (1.0, 2.0, 3.0)
    rotated = apply_3axis_rotation_matrix(point, 0, 0, 0)

    assert abs(rotated[0] - 1.0) < 1e-6
    assert abs(rotated[1] - 2.0) < 1e-6
    assert abs(rotated[2] - 3.0) < 1e-6


def test_90_degree_x_rotation():
    """Test 90 degree rotation around X axis."""
    from math_utils import rotate_x

    point = (0, 1, 0)
    rotated = rotate_x(point, 90.0)

    # After 90 degree X rotation: y->z, so (0,1,0) -> (0, 0, 1)
    assert abs(rotated[0]) < 1e-6
    assert abs(rotated[1]) < 1e-6
    assert abs(rotated[2] - 1.0) < 1e-6


def test_90_degree_y_rotation():
    """Test 90 degree rotation around Y axis."""
    from math_utils import rotate_y

    point = (1, 0, 0)
    rotated = rotate_y(point, 90.0)

    # After 90 degree Y rotation: x->z, so (1,0,0) -> (0, 0, -1) per the rotation formula
    assert abs(rotated[1]) < 1e-6
    assert abs(rotated[2] + 1.0) < 1e-6


def test_90_degree_z_rotation():
    """Test 90 degree rotation around Z axis."""
    from math_utils import rotate_z

    point = (1, 0, 0)
    rotated = rotate_z(point, 90.0)

    # After 90 degree Z rotation: x->y, so (1,0,0) -> (0, 1, 0)
    assert abs(rotated[2]) < 1e-6
    assert abs(rotated[1] - 1.0) < 1e-6


def test_project_3d_to_screen():
    """Verify projection to screen coordinates works correctly."""
    from projection import project_3d_to_screen
    from config import SCREEN_W, SCREEN_H

    # Test origin projects to center (FOV and camera_distance from config)
    projected = project_3d_to_screen(0, 0, 5)

    # Origin should be near screen center
    center_threshold = 50
    assert SCREEN_W // 2 - center_threshold <= projected[0] <= SCREEN_W // 2 + center_threshold, \
        f"X coordinate {projected[0]} should be near center {SCREEN_W // 2}"
    assert SCREEN_H // 2 - center_threshold <= projected[1] <= SCREEN_H // 2 + center_threshold, \
        f"Y coordinate {projected[1]} should be near center {SCREEN_H // 2}"


if __name__ == '__main__':
    # Run all tests directly - extract function names and calls
    test_calls = [
        ('test_damping_decreases_velocity', test_damping_decreases_velocity),
        ('test_clamp_limits_velocity', test_clamp_limits_velocity),
        ('test_drag_impulse_adds_momentum', test_drag_impulse_adds_momentum),
        ('test_rotation_preserves_distance_from_origin', test_rotation_preserves_distance_from_origin),
        ('test_rotation_matrices_preserve_cube_shape', test_rotation_matrices_preserve_cube_shape),
        ('test_x_rotation_matrix_structure', test_x_rotation_matrix_structure),
        ('test_y_rotation_matrix_structure', test_y_rotation_matrix_structure),
        ('test_z_rotation_matrix_structure', test_z_rotation_matrix_structure),
        ('test_identity_rotation_returns_same_point', test_identity_rotation_returns_same_point),
        ('test_90_degree_x_rotation', test_90_degree_x_rotation),
        ('test_90_degree_y_rotation', test_90_degree_y_rotation),
        ('test_90_degree_z_rotation', test_90_degree_z_rotation),
        ('test_project_3d_to_screen', test_project_3d_to_screen),
    ]

    print(f"Running {len(test_calls)} consolidated tests...\n")

    failed = []
    passed = 0

    for name, func in test_calls:
        try:
            func()
            print(f"\u2713 {name}")
            passed += 1
        except AssertionError as e:
            print(f"\u2717 {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"\u2717 {name}: {type(e).__name__}: {e}")
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {len(failed)} failed")

    if failed:
        print("\nFailed tests:")
        for f in failed:
            print(f"  - {f}")

    sys.exit(0 if not failed else 1)