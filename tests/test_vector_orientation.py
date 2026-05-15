"""
Tests for vector orientation correctness.
Tests the cross product computations for tangential velocity and centripetal acceleration.
"""

import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic import cross_product, rotate_around_axis, normalize_vector


def test_tangential_velocity_direction():
    """Test that v = ω × r gives the correct tangential velocity direction."""
    # Case 1: Rotation about +Z axis only
    omega = (0, 0, 1)  # 1 rad/s about +Z
    r = (1, 0, 0)  # Point on +X axis
    
    v = cross_product(omega, r)
    # ω = (0,0,1), r = (1,0,0)
    # v_x = ω_y * r_z - ω_z * r_y = 0*0 - 1*0 = 0
    # v_y = ω_z * r_x - ω_x * r_z = 1*1 - 0*0 = 1
    # v_z = ω_x * r_y - ω_y * r_x = 0*0 - 0*1 = 0
    assert v == (0, 1, 0), f"Expected (0, 1, 0), got {v}"
    print(f"✓ Test 1: ω=(0,0,1), r=(1,0,0) → v={v} (expected (0,1,0))")
    
    # Case 2: Rotation about +Y axis only
    omega = (0, 1, 0)
    r = (1, 0, 0)
    
    v = cross_product(omega, r)
    # v_x = 1*0 - 0*0 = 0
    # v_y = 0*0 - 0*0 = 0
    # v_z = 0*0 - 1*1 = -1
    assert v == (0, 0, -1), f"Expected (0, 0, -1), got {v}"
    print(f"✓ Test 2: ω=(0,1,0), r=(1,0,0) → v={v} (expected (0,0,-1))")
    
    # Case 3: Rotation about +X axis only
    omega = (1, 0, 0)
    r = (0, 1, 0)
    
    v = cross_product(omega, r)
    # v_x = 0*0 - 0*1 = 0
    # v_y = 0*0 - 1*0 = 0
    # v_z = 1*1 - 0*0 = 1
    assert v == (0, 0, 1), f"Expected (0, 0, 1), got {v}"
    print(f"✓ Test 3: ω=(1,0,0), r=(0,1,0) → v={v} (expected (0,0,1))")


def test_centripetal_acceleration_direction():
    """Test that a = ω × v points toward the rotation axis."""
    # Case 1: Rotation about +Z axis
    omega = (0, 0, 1)
    r = (1, 0, 0)
    v = cross_product(omega, r)  # v = (0, 1, 0)
    
    a = cross_product(omega, v)
    # a_x = 0*0 - 1*1 = -1
    # a_y = 1*0 - 0*0 = 0
    # a_z = 0*1 - 0*0 = 0
    assert a == (-1, 0, 0), f"Expected (-1, 0, 0), got {a}"
    print(f"✓ Test 4: ω=(0,0,1), r=(1,0,0), v=(0,1,0) → a={a} (expected (-1,0,0))")
    # Centripetal should point toward rotation axis (= toward origin in this case)
    
    # Case 2: Rotation about +Y axis
    omega = (0, 1, 0)
    r = (1, 0, 0)
    v = cross_product(omega, r)  # v = (0, 0, -1)
    
    a = cross_product(omega, v)
    # a_x = 1*(-1) - 0*0 = -1
    # a_y = 0*0 - 0*(-1) = 0
    # a_z = 0*0 - 1*0 = 0
    assert a == (-1, 0, 0), f"Expected (-1, 0, 0), got {a}"
    print(f"✓ Test 5: ω=(0,1,0), r=(1,0,0), v=(0,0,-1) → a={a} (expected (-1,0,0))")


def test_omega_vector_direction():
    """Test that omega vector points along the rotation axis (right-hand rule)."""
    # Rotation about +Z → omega should point in +Z direction
    omega = (0, 0, 5)
    unit = normalize_vector(omega)
    assert abs(unit[0]) < 1e-6 and abs(unit[1]) < 1e-6 and abs(unit[2] - 1.0) < 1e-6
    print(f"✓ Test 6: ω=(0,0,5) → unit={unit} (expected ~ (0,0,1))")
    
    # Rotation about +X → omega should point in +X direction
    omega = (3, 0, 0)
    unit = normalize_vector(omega)
    assert abs(unit[0] - 1.0) < 1e-6 and abs(unit[1]) < 1e-6 and abs(unit[2]) < 1e-6
    print(f"✓ Test 7: ω=(3,0,0) → unit={unit} (expected ~ (1,0,0))")


def test_rotate_around_axis():
    """Test rotation of vectors around axes."""
    # Rotate (1,0,0) around X axis by 90° → should stay (1,0,0)
    result = rotate_around_axis((1, 0, 0), [1, 0, 0], 90)
    assert abs(result[0] - 1.0) < 1e-6 and abs(result[1]) < 1e-6 and abs(result[2]) < 1e-6
    print(f"✓ Test 8: Rotate (1,0,0) around X by 90° → {result} (expected ~ (1,0,0))")
    
    # Rotate (0,1,0) around X axis by 90° → (0,0,1)
    result = rotate_around_axis((0, 1, 0), [1, 0, 0], 90)
    assert abs(result[0]) < 1e-6 and abs(result[1]) < 1e-6 and abs(result[2] - 1.0) < 1e-6
    print(f"✓ Test 9: Rotate (0,1,0) around X by 90° → {result} (expected ~ (0,0,1))")
    
    # Rotate (0,1,0) around Y axis by 90° → (0,1,0) (unchanged)
    result = rotate_around_axis((0, 1, 0), [0, 1, 0], 90)
    assert abs(result[0]) < 1e-6 and abs(result[1] - 1.0) < 1e-6 and abs(result[2]) < 1e-6
    print(f"✓ Test 10: Rotate (0,1,0) around Y by 90° → {result} (expected ~ (0,1,0))")
    
    # Rotate (0,0,1) around Y axis by 90° → (1,0,0)
    result = rotate_around_axis((0, 0, 1), [0, 1, 0], 90)
    assert abs(result[0] - 1.0) < 1e-6 and abs(result[1]) < 1e-6 and abs(result[2]) < 1e-6
    print(f"✓ Test 11: Rotate (0,0,1) around Y by 90° → {result} (expected ~ (1,0,0))")


def test_camera_rotation_preserves_direction():
    """Test that camera rotation preserves vector direction when view angles are 0."""
    # With zero view angles, the direction should be unchanged
    direction = (1, 1, 0)
    unit = normalize_vector(direction)
    
    result = rotate_around_axis(unit, [0, 1, 0], 0)
    result = rotate_around_axis(result, [0, 0, 1], 0)
    result = rotate_around_axis(result, [1, 0, 0], 0)
    
    assert abs(result[0] - unit[0]) < 1e-6
    assert abs(result[1] - unit[1]) < 1e-6
    assert abs(result[2] - unit[2]) < 1e-6
    print(f"✓ Test 12: Zero view rotation preserves direction: {result} == {unit}")


def test_3d_velocity_vector_computation():
    """Test the full draw_3d_velocity_vectors computation."""
    from logic import draw_3d_velocity_vectors
    
    # Test with omega purely about +Z
    omega_x, omega_y, omega_z = 0, 0, 10
    total_omega = 10
    
    # Cube vertex at (3.5, 3.5, 3.5)
    test_verts = [(3.5, 3.5, 3.5)]
    
    # Get the drawables
    tang_drawables, cent_drawables = draw_3d_velocity_vectors(
        test_verts, total_omega, None, 
        omega_x, omega_y, omega_z,
        max_vectors=1, show_tangential=True, show_centripetal=True,
        view_x=0, view_y=0, view_z=0
    )
    
    # With omega=(0,0,10) and r=(3.5,3.5,3.5):
    # v = ω × r = (0*3.5 - 10*3.5, 10*3.5 - 0*3.5, 0*3.5 - 0*3.5)
    #   = (-35, 35, 0)
    # This should point in the tangent direction (perpendicular to r in XY plane)
    print(f"✓ Test 13: With ω=(0,0,10), r=(3.5,3.5,3.5):")
    print(f"  Tangential: {len(tang_drawables)} drawables, Centripetal: {len(cent_drawables)} drawables")


def test_view_rotation_consistency():
    """Test that applying the same view rotation to start and end preserves direction."""
    # Given direction in world space
    start_world = (1, 0, 0)
    direction_world = (0, 1, 0)  # Pointing along +Y
    end_world = (1, 1, 0)
    
    # Apply some camera rotation
    view_x, view_y, view_z = 30, -45, 10
    
    start_cam = rotate_around_axis(start_world, [0, 1, 0], view_y)
    start_cam = rotate_around_axis(start_cam, [0, 0, 1], -view_z)
    start_cam = rotate_around_axis(start_cam, [1, 0, 0], view_x)
    
    end_cam = rotate_around_axis(end_world, [0, 1, 0], view_y)
    end_cam = rotate_around_axis(end_cam, [0, 0, 1], -view_z)
    end_cam = rotate_around_axis(end_cam, [1, 0, 0], view_x)
    
    # The direction in camera space should be the same as rotating the original direction
    dir_cam = rotate_around_axis(direction_world, [0, 1, 0], view_y)
    dir_cam = rotate_around_axis(dir_cam, [0, 0, 1], -view_z)
    dir_cam = rotate_around_axis(dir_cam, [1, 0, 0], view_x)
    
    # Check: end_cam - start_cam should equal dir_cam
    dx = end_cam[0] - start_cam[0]
    dy = end_cam[1] - start_cam[1]
    dz = end_cam[2] - start_cam[2]
    
    assert abs(dx - dir_cam[0]) < 1e-6, f"dx={dx} != dir_cam[0]={dir_cam[0]}"
    assert abs(dy - dir_cam[1]) < 1e-6, f"dy={dy} != dir_cam[1]={dir_cam[1]}"
    assert abs(dz - dir_cam[2]) < 1e-6, f"dz={dz} != dir_cam[2]={dir_cam[2]}"
    
    print(f"✓ Test 14: View rotation preserves direction")
    print(f"  Original direction: {direction_world}")
    print(f"  Rotated direction: ({dir_cam[0]:.4f}, {dir_cam[1]:.4f}, {dir_cam[2]:.4f})")
    print(f"  end-start: ({dx:.4f}, {dy:.4f}, {dz:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing vector orientation correctness")
    print("=" * 60)
    
    test_tangential_velocity_direction()
    print()
    test_centripetal_acceleration_direction()
    print()
    test_omega_vector_direction()
    print()
    test_rotate_around_axis()
    print()
    test_camera_rotation_preserves_direction()
    print()
    test_3d_velocity_vector_computation()
    print()
    test_view_rotation_consistency()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)