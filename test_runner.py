"""
Unit Test Runner for Angular Velocity Demo

Provides synthetic drag input and frame-based testing without pygame GUI.
Run all tests: python angular_velocity/test_runner.py
Run specific test categories:
  python angular_velocity/test_runner.py --run-geometry-tests
  python angular_velocity/test_runner.py --run-physics-tests
  python angular_velocity/test_runner.py --run-math-tests
"""

import argparse
import sys
import os

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import math
import numpy as np
from logic import (
    # Pure math functions
    cross_product, dot_product, vector_magnitude, normalize_vector,
    rotate_around_axis, rotate_x, rotate_y, rotate_z,
    apply_3axis_rotation_matrix, project_3d_to_screen,
    # Cube geometry
    CUBE_VERTS, CUBE_EDGES, CUBE_FACE_INDICES, compute_face_center,
    # Physics functions
    apply_drag_impulse, apply_drag_impulse_damping, clamp_angular_velocity,
    # Config constants
    SCREEN_W, SCREEN_H, CUBE_SIZE, FRAME_RATE, CONFIG, COLORS
)


def assert_almost_equal(a, b, tol=1e-6):
    """Assertion helper that prints error details on failure."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        assert len(a) == len(b), "Tuple lengths differ"
        for i in range(len(a)):
            assert abs(a[i] - b[i]) < tol, f"Expected {b[i]}, got {a[i]} (diff: {abs(a[i]-b[i]):.2e}) at index {i} in {a} vs {b}"
    else:
        assert abs(a - b) < tol, f"Expected {b}, got {a} (diff: {abs(a-b):.2e})"


def run_geometry_tests():
    """Test cube geometry and vertex definitions."""
    print("\n=== GEOMETRY TESTS ===")
    
    # Test CUBE_VERTS shape and range
    assert CUBE_VERTS.shape == (8, 3), f"CUBE_VERTS shape mismatch: {CUBE_VERTS.shape}"
    
    for i, v in enumerate(CUBE_VERTS):
        assert abs(v[0]) <= CUBE_SIZE * (1 + 1e-6), f"Vertex {i} X out of bounds"
        assert abs(v[1]) <= CUBE_SIZE * (1 + 1e-6), f"Vertex {i} Y out of bounds"
        assert abs(v[2]) <= CUBE_SIZE * (1 + 1e-6), f"Vertex {i} Z out of bounds"
    
    print("✓ Cube vertices shape and range verified")
    
    # Test CUBE_EDGES connect valid vertices
    for edge in CUBE_EDGES:
        i, j = edge
        assert 0 <= i < 8, f"Edge vertex {i} out of range"
        assert 0 <= j < 8, f"Edge vertex {j} out of range"
    
    print("✓ Cube edges reference valid vertices")
    
    # Test face indices have correct number of vertices (quads)
    for face in CUBE_FACE_INDICES:
        assert len(face) == 4, f"Face has {len(face)} vertices, expected 4"
    
    print("✓ All faces are quadrilaterals")
    
    # Test compute_face_center
    face0_center = compute_face_center(CUBE_FACE_INDICES[0])  # Front face
    # The expected_center should be derived from the actual vertex positions
    # Let's re-calculate expected based on CUBE_VERTS for robustness
    expected_center_calc = np.mean([CUBE_VERTS[i] for i in CUBE_FACE_INDICES[0]], axis=0)
    assert_almost_equal(face0_center[0], expected_center_calc[0])
    assert_almost_equal(face0_center[1], expected_center_calc[1])
    assert_almost_equal(face0_center[2], expected_center_calc[2])
    print("✓ Face center computation verified")
    
    return True


def run_math_tests():
    """Test pure math functions."""
    print("\n=== MATH TESTS ===")
    
    # Test cross_product
    a = (1, 0, 0)
    b = (0, 1, 0)
    result = cross_product(a, b)
    assert_almost_equal(result, (0, 0, 1), tol=1e-6), "Cross product X×Y≠Z"
    
    a = (0, 1, 0)
    b = (0, 0, 1)
    result = cross_product(a, b)
    assert_almost_equal(result, (1, 0, 0), tol=1e-6), "Cross product Y×Z≠X"
    
    a = (0, 0, 1)
    b = (1, 0, 0)
    result = cross_product(a, b)
    assert_almost_equal(result, (0, 1, 0), tol=1e-6), "Cross product Z×X≠Y"
    
    print("✓ Cross product verified (cyclic permutations)")
    
    # Test dot_product
    a = (1, 2, 3)
    b = (4, 5, 6)
    expected = 1*4 + 2*5 + 3*6
    result = dot_product(a, b)
    assert_almost_equal(result, expected), f"Dot product: {result}≠{expected}"
    
    # Orthogonal vectors have dot product of zero
    a = (1, 0, 0)
    b = (0, 1, 0)
    result = dot_product(a, b)
    assert_almost_equal(result, 0), "Orthogonal vectors should have dot=0"
    
    print("✓ Dot product verified")
    
    # Test vector_magnitude
    v = (3, 4, 0)
    expected_mag = math.sqrt(9 + 16)
    result = vector_magnitude(v)
    assert_almost_equal(result, expected_mag), f"Magnitude: {result}≠{expected_mag}"
    
    print("✓ Vector magnitude verified")
    
    # Test normalize_vector
    v = (3, 4, 0)
    expected_norm = (0.6, 0.8, 0.0)
    result = normalize_vector(v)
    assert_almost_equal(result, expected_norm, tol=1e-6), f"Normalized: {result}≠{expected_norm}"
    
    # Zero vector normalization
    v = (0, 0, 0)
    result = normalize_vector(v)
    assert result == (0, 0, 0), "Zero vector should normalize to (0,0,0)"
    
    print("✓ Vector normalization verified")
    
    # Test rotation functions
    p = (1, 0, 0)
    r90x = rotate_x(p, 90)
    assert_almost_equal(r90x, (1.0, 0.0, 0.0), tol=1e-6), "X-rotation of (1,0,0) by 90 deg should be (1,0,0)"
    
    p = (0, 1, 0)
    r90x = rotate_x(p, 90)
    assert_almost_equal(r90x, (0.0, 0.0, 1.0), tol=1e-6), "X-rotation of (0,1,0) by 90 deg should be (0,0,1)"

    p = (0, 0, 1)
    r90x = rotate_x(p, 90)
    assert_almost_equal(r90x, (0.0, -1.0, 0.0), tol=1e-6), "X-rotation of (0,0,1) by 90 deg should be (0,-1,0)"

    p = (0, 1, 0)
    r90y = rotate_y(p, 90)
    assert_almost_equal(r90y, (0.0, 1.0, 0.0), tol=1e-6), "Y-rotation of (0,1,0) by 90 deg should be (0,1,0)"

    p = (1, 0, 0)
    r90y = rotate_y(p, 90)
    assert_almost_equal(r90y, (0.0, 0.0, -1.0), tol=1e-6), "Y-rotation of (1,0,0) by 90 deg should be (0,0,-1)"

    p = (0, 0, 1)
    r90y = rotate_y(p, 90)
    assert_almost_equal(r90y, (1.0, 0.0, 0.0), tol=1e-6), "Y-rotation of (0,0,1) by 90 deg should be (1,0,0)"

    p = (0, 0, 1)
    r90z = rotate_z(p, 90)
    assert_almost_equal(r90z, (0.0, 0.0, 1.0), tol=1e-6), "Z-rotation of (0,0,1) by 90 deg should be (0,0,1)"

    p = (1, 0, 0)
    r90z = rotate_z(p, 90)
    assert_almost_equal(r90z, (0.0, 1.0, 0.0), tol=1e-6), "Z-rotation of (1,0,0) by 90 deg should be (0,1,0)"

    p = (0, 1, 0)
    r90z = rotate_z(p, 90)
    assert_almost_equal(r90z, (-1.0, 0.0, 0.0), tol=1e-6), "Z-rotation of (0,1,0) by 90 deg should be (-1,0,0)"
    
    print("✓ Single-axis rotations verified")
    
    # Test arbitrary axis rotation (around X axis with unit vector [1,0,0])
    p = (0, 1, 0)
    result = rotate_around_axis(p, [1, 0, 0], 90)
    assert_almost_equal(result, (0.0, 0.0, 1.0), tol=1e-6), "Arbitrary X-axis rotation of (0,1,0) by 90 deg should be (0,0,1)"

    p = (0, 0, 1)
    result = rotate_around_axis(p, [0, 1, 0], 90)
    assert_almost_equal(result, (1.0, 0.0, 0.0), tol=1e-6), "Arbitrary Y-axis rotation of (0,0,1) by 90 deg should be (1,0,0)"

    p = (1, 0, 0)
    result = rotate_around_axis(p, [0, 0, 1], 90)
    assert_almost_equal(result, (0.0, 1.0, 0.0), tol=1e-6), "Arbitrary Z-axis rotation of (1,0,0) by 90 deg should be (0,1,0)"
    
    print("✓ Arbitrary axis rotation verified")
    
    return True


def run_projection_tests():
    """Test 3D to screen projection."""
    print("\n=== PROJECTION TESTS ===")
    
    center_x = SCREEN_W // 2
    center_y = SCREEN_H // 2
    
    # Point at origin projects to screen center
    result = project_3d_to_screen(0, 0, 0)
    assert result == (center_x, center_y), f"Origin should project to ({center_x}, {center_y})"
    
    print("✓ Origin projection verified")
    
    # Point along X axis (right with positive scale_x)
    result = project_3d_to_screen(10, 0, 0)
    assert result[0] > center_x, "Point right of origin should have larger X"
    assert abs(result[1] - center_y) < 2, "Y should be at center"
    
    print("✓ X-axis projection verified")
    
    # Point along Y axis: with negative scale_y, positive Y goes UP (screen coords), not down
    result = project_3d_to_screen(0, 10, 0)
    assert abs(result[0] - center_x) < 2, "X should be at center"
    # With scale_y = -42.0, py = 360 + 10 * (-42) = 360 - 420 = -60 (above center)
    assert result[1] < center_y, f"With negative scale_y, positive Y should go UP on screen: {result} vs center ({center_x}, {center_y})"
    
    # Point along Z axis (depth toward viewer)
    result = project_3d_to_screen(0, 0, 10)
    assert abs(result[0] - center_x) < 2, "Z offset should have negligible X shift"
    assert abs(result[1] - center_y) < 2, "Z offset should have negligible Y shift"
    
    print("✓ Axis projections verified")
    
    return True


def run_physics_tests():
    """Test physics functions."""
    print("\n=== PHYSICS TESTS ===")
    
    # Test apply_drag_impulse with synthetic_test config
    synth_config = CONFIG.get('synthetic_test', {})
    
    omega_x, omega_y, omega_z = 0.0, 0.0, 0.0
    
    # Apply a drag impulse
    delta_x, delta_y = 1.0, 0.5
    omega_x, omega_y, omega_z = apply_drag_impulse(omega_x, omega_y, omega_z, delta_x, delta_y)
    
    expected_x = synth_config.get('drag_delta_omega_x', 0.1) * delta_x
    expected_y = synth_config.get('drag_delta_omega_y', 0.08) * delta_y
    expected_z_offset = synth_config.get('drag_delta_omega_z_offset', 0.05) * delta_x
    expected_z_base = synth_config.get('drag_delta_omega_z_base', 0.03) * delta_y
    
    assert_almost_equal(omega_x, expected_x, tol=1e-5), f"Delta omega X: {omega_x}≠{expected_x}"
    assert_almost_equal(omega_y, expected_y, tol=1e-5), f"Delta omega Y: {omega_y}≠{expected_y}"
    
    # Z component is sum of offset and base
    expected_z = expected_z_offset + expected_z_base
    assert_almost_equal(omega_z, expected_z, tol=1e-5), f"Delta omega Z: {omega_z}≠{expected_z}"
    
    print("✓ Drag impulse application verified")
    
    # Test apply_drag_impulse_damping (no input damping)
    result = apply_drag_impulse_damping(0.1, 0.2, 0.3)
    expected_damping = 0.985
    
    assert_almost_equal(result[0], 0.1 * expected_damping, tol=1e-5), "Input damping failed"
    assert_almost_equal(result[1], 0.2 * expected_damping, tol=1e-5)
    assert_almost_equal(result[2], 0.3 * expected_damping, tol=1e-5)
    
    print("✓ Drag impulse damping verified")
    
    # Test velocity-dependent damping
    result = apply_drag_impulse_damping(0.5, 0.8, 0.4)
    vec_mag = vector_magnitude((0.5, 0.8, 0.4))
    speed_factor = min(vec_mag / CONFIG['angular_velocity']['max_speed'], 1.0)
    expected_damping = expected_damping + (speed_factor * 0.01)
    
    assert_almost_equal(result[0], 0.5 * expected_damping, tol=1e-4), "Velocity-dependent damping failed"
    
    print("✓ Velocity-dependent damping verified")
    
    # Test clamp_angular_velocity
    result = clamp_angular_velocity(0.1, 0.2, 0.3)
    total_mag = vector_magnitude((0.1, 0.2, 0.3))
    max_allowed = CONFIG['angular_velocity']['max_speed'] * 1.5
    
    # Should not change if below max
    expected_scale = min(total_mag / max_allowed, 1.0)
    assert_almost_equal(result[0], 0.1 * expected_scale, tol=1e-6), "Clamping failed (no clamping needed)"
    
    print("✓ Angular velocity clamping verified")
    
    # Test extreme case: apply drag repeatedly until clamping kicks in
    omega_x, omega_y, omega_z = 0.0, 0.0, 0.0
    
    for _ in range(100):
        omega_x, omega_y, omega_z = apply_drag_impulse(omega_x, omega_y, omega_z, 1.0, 1.0)
        omega_x, omega_y, omega_z = apply_drag_impulse_damping(omega_x, omega_y, omega_z)
    
    # After many impulses, should be clamped
    total_mag = vector_magnitude((omega_x, omega_y, omega_z))
    max_allowed = CONFIG['angular_velocity']['max_speed'] * 1.5
    assert total_mag <= max_allowed * (1 + 1e-6), f"Angular velocity unclamped: {total_mag} > {max_allowed}"
    
    print("✓ Extreme case clamping verified")
    
    return True


def test_synthetic_drag_sequence():
    """Test that synthetic drag input sequences work correctly."""
    print("\n=== SYNTHETIC DRAG SEQUENCE TEST ===")
    
    # Test applying a sequence of drag inputs
    omega_x, omega_y, omega_z = 0.0, 0.0, 0.0
    
    # Simulate a mouse drag: move right and down
    for _ in range(10):
        omega_x, omega_y, omega_z = apply_drag_impulse(omega_x, omega_y, omega_z, 5, 3)
        omega_x, omega_y, omega_z = apply_drag_impulse_damping(omega_x, omega_y, omega_z)
    
    # After 10 steps with input (5, 3):
    # Expected: approximately 1.0 * 10 + small damping losses
    synth_config = CONFIG.get('synthetic_test', {})
    expected_x = synth_config.get('drag_delta_omega_x', 0.1) * 5 * 10
    expected_y = synth_config.get('drag_delta_omega_y', 0.08) * 3 * 10
    
    assert omega_x > expected_x * 0.9, f"Synthetic drag X too small: {omega_x} < {expected_x}"
    assert omega_y > expected_y * 0.9, f"Synthetic drag Y too small: {omega_y} < {expected_y}"
    
    print("✓ Synthetic drag sequence verified")


def run_all_tests():
    """Run all test categories."""
    print("=" * 60)
    print("ANGULAR VELOCITY DEMO - UNIT TEST SUITE")
    print("=" * 60)
    
    try:
        results = []
        
        if run_geometry_tests():
            results.append(("Geometry Tests", True))
        else:
            results.append(("Geometry Tests", False))
        
        if run_math_tests():
            results.append(("Math Tests", True))
        else:
            results.append(("Math Tests", False))
        
        if run_projection_tests():
            results.append(("Projection Tests", True))
        else:
            results.append(("Projection Tests", False))
        
        if run_physics_tests():
            results.append(("Physics Tests", True))
        else:
            results.append(("Physics Tests", False))
        
        # Add synthetic drag sequence test
        test_synthetic_drag_sequence()
        results.append(("Synthetic Drag Sequence", True))
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for name, result in results:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status}: {name}")
        
        print("-" * 40)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✓ ALL TESTS PASSED!")
            return True
        else:
            print(f"\n✗ {total - passed} test(s) failed")
            return False
            
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='Angular Velocity Demo Unit Tests')
    parser.add_argument('--run-geometry-tests', action='store_true', help='Run geometry tests only')
    parser.add_argument('--run-math-tests', action='store_true', help='Run math tests only')
    parser.add_argument('--run-projection-tests', action='store_true', help='Run projection tests only')
    parser.add_argument('--run-physics-tests', action='store_true', help='Run physics tests only')
    parser.add_argument('--synthetic-drag', action='store_true', help='Run synthetic drag sequence test only')
    
    args = parser.parse_args()
    
    if any([args.run_geometry_tests, args.run_math_tests, 
            args.run_projection_tests, args.run_physics_tests, 
            args.synthetic_drag]):
        # Run only requested tests
        passed = True
        
        if args.run_geometry_tests and not run_geometry_tests():
            passed = False
        if args.run_math_tests and not run_math_tests():
            passed = False
        if args.run_projection_tests and not run_projection_tests():
            passed = False
        if args.run_physics_tests and not run_physics_tests():
            passed = False
        if args.synthetic_drag and not test_synthetic_drag_sequence():
            passed = False
        
        sys.exit(0 if passed else 1)
    else:
        # Run all tests by default
        success = run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()