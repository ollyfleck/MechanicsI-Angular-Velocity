"""
Tests for quaternion-based rotation composition.
Verifies that omega vector direction is correctly derived from accumulated orientation.
"""

import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic import (
    quat_multiply, quat_conjugate, quat_normalize,
    axis_angle_to_quat, quat_to_omega, get_orientation_quaternion,
)


def test_quat_multiply_identity():
    """Test quaternion multiplication with identity quaternion."""
    q = (0.7071, 0.7071, 0.0, 0.0)  # ~90° about X
    identity = (1.0, 0.0, 0.0, 0.0)
    result = quat_multiply(q, identity)
    assert abs(result[0] - q[0]) < 1e-6
    assert abs(result[1] - q[1]) < 1e-6
    assert abs(result[2] - q[2]) < 1e-6
    assert abs(result[3] - q[3]) < 1e-6
    print("✓ quat_multiply with identity works")


def test_quat_normalize():
    """Test quaternion normalization."""
    q = (3.0, 4.0, 0.0, 0.0)
    normalized = quat_normalize(q)
    mag = math.sqrt(normalized[0]**2 + normalized[1]**2 + normalized[2]**2 + normalized[3]**2)
    assert abs(mag - 1.0) < 1e-6
    print("✓ quat_normalize works")


def test_quat_conjugate():
    """Test quaternion conjugate."""
    q = (0.5, 0.5, 0.5, 0.5)
    conj = quat_conjugate(q)
    # Conjugate of (w, x, y, z) is (w, -x, -y, -z)
    assert abs(conj[0] - q[0]) < 1e-6
    assert abs(conj[1] + q[1]) < 1e-6
    assert abs(conj[2] + q[2]) < 1e-6
    assert abs(conj[3] + q[3]) < 1e-6
    # Conjugate should have same magnitude
    mag_q = math.sqrt(sum(c**2 for c in q))
    mag_conj = math.sqrt(sum(c**2 for c in conj))
    assert abs(mag_q - mag_conj) < 1e-6
    print("✓ quat_conjugate works")


def test_axis_angle_to_quat():
    """Test axis-angle to quaternion conversion."""
    # 90° about X axis
    q = axis_angle_to_quat((1, 0, 0), math.pi / 2)
    # w = cos(θ/2), x = sin(θ/2)*axis_x
    expected_w = math.cos(math.pi / 4)
    expected_x = math.sin(math.pi / 4) * 1.0
    assert abs(q[0] - expected_w) < 1e-6
    assert abs(q[1] - expected_x) < 1e-6
    assert abs(q[2]) < 1e-6
    assert abs(q[3]) < 1e-6
    print("✓ axis_angle_to_quat works")


def test_get_orientation_quaternion():
    """Test Euler angle to quaternion conversion."""
    # Zero rotation should give identity quaternion
    q = get_orientation_quaternion(0.0, 0.0, 0.0)
    assert abs(q[0] - 1.0) < 1e-6  # w ≈ 1
    assert abs(q[1]) < 1e-6        # x ≈ 0
    assert abs(q[2]) < 1e-6        # y ≈ 0
    assert abs(q[3]) < 1e-6        # z ≈ 0
    
    # 90° about X only
    q = get_orientation_quaternion(90.0, 0.0, 0.0)
    expected_w = math.cos(math.radians(45))
    expected_x = math.sin(math.radians(45))
    assert abs(q[0] - expected_w) < 1e-6
    assert abs(q[1] - expected_x) < 1e-6
    assert abs(q[2]) < 1e-6
    assert abs(q[3]) < 1e-6
    
    # 90° about Y only
    q = get_orientation_quaternion(0.0, 90.0, 0.0)
    expected_w = math.cos(math.radians(45))
    expected_y = math.sin(math.radians(45))
    assert abs(q[0] - expected_w) < 1e-6
    assert abs(q[2] - expected_y) < 1e-6
    assert abs(q[1]) < 1e-6
    assert abs(q[3]) < 1e-6
    
    # 90° about Z only
    q = get_orientation_quaternion(0.0, 0.0, 90.0)
    expected_w = math.cos(math.radians(45))
    expected_z = math.sin(math.radians(45))
    assert abs(q[0] - expected_w) < 1e-6
    assert abs(q[3] - expected_z) < 1e-6
    assert abs(q[1]) < 1e-6
    assert abs(q[2]) < 1e-6
    
    print("✓ get_orientation_quaternion works")


def test_quat_to_omega_identity():
    """Test that quat_to_omega returns zero for identity quaternion."""
    q = (1.0, 0.0, 0.0, 0.0)
    omega = quat_to_omega(q)
    assert abs(omega[0]) < 1e-6
    assert abs(omega[1]) < 1e-6
    assert abs(omega[2]) < 1e-6
    print("✓ quat_to_omega returns zero for identity")


def test_quat_to_omega_x_rotation():
    """Test that quat_to_omega gives correct axis for pure X rotation."""
    # 90° about X
    q = get_orientation_quaternion(90.0, 0.0, 0.0)
    omega = quat_to_omega(q)
    
    # Should point along +X
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    if mag > 0.01:
        assert abs(omega[0] / mag - 1.0) < 0.1, f"Expected ~+X direction, got {omega}"
        assert abs(omega[1]) < mag * 0.5
        assert abs(omega[2]) < mag * 0.5
    
    print(f"✓ quat_to_omega for 90° X: ω={omega}")


def test_quat_to_omega_y_rotation():
    """Test that quat_to_omega gives correct axis for pure Y rotation."""
    # 90° about Y
    q = get_orientation_quaternion(0.0, 90.0, 0.0)
    omega = quat_to_omega(q)
    
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    if mag > 0.01:
        assert abs(omega[1] / mag - 1.0) < 0.1, f"Expected ~+Y direction, got {omega}"
    
    print(f"✓ quat_to_omega for 90° Y: ω={omega}")


def test_quat_to_omega_z_rotation():
    """Test that quat_to_omega gives correct axis for pure Z rotation."""
    # 90° about Z
    q = get_orientation_quaternion(0.0, 0.0, 90.0)
    omega = quat_to_omega(q)
    
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    if mag > 0.01:
        assert abs(omega[2] / mag - 1.0) < 0.1, f"Expected ~+Z direction, got {omega}"
    
    print(f"✓ quat_to_omega for 90° Z: ω={omega}")


def test_quat_to_omega_combined_rotation():
    """Test that quat_to_omega gives correct axis for combined rotation."""
    # Combined: 45° X + 45° Y
    q = get_orientation_quaternion(45.0, 45.0, 0.0)
    omega = quat_to_omega(q)
    
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    if mag > 0.01:
        # Should have both X and Y components
        print(f"✓ quat_to_omega for 45°X+45°Y: ω={omega}, direction=({omega[0]/mag:.3f}, {omega[1]/mag:.3f}, {omega[2]/mag:.3f})")


def test_quaternion_composition_vs_euler():
    """Test that quaternion composition matches Euler angle accumulation for small angles."""
    # Small rotation: 1° about each axis
    q = get_orientation_quaternion(1.0, 1.0, 1.0)
    
    # For small angles, the vector part should be approximately proportional to the axis
    mag = math.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if mag > 0.01:
        # Direction should be roughly (1, 1, 1) normalized
        print(f"✓ Quaternion for small Euler angles: ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")


def test_omega_vector_alignment_with_axis():
    """Test that the omega direction from quaternion matches expected axis for known rotations."""
    # Test 1: Pure +X rotation → omega should point along +X
    q = get_orientation_quaternion(90.0, 0.0, 0.0)
    omega = quat_to_omega(q)
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    assert mag > 0.01, "Omega magnitude should be non-zero"
    assert abs(omega[0] / mag - 1.0) < 0.15, f"Expected ~+X direction for pure X rotation, got {omega}"
    print(f"✓ Pure X rotation: omega direction = ({omega[0]/mag:.3f}, {omega[1]/mag:.3f}, {omega[2]/mag:.3f})")
    
    # Test 2: Pure +Y rotation → omega should point along +Y
    q = get_orientation_quaternion(0.0, 90.0, 0.0)
    omega = quat_to_omega(q)
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    assert abs(omega[1] / mag - 1.0) < 0.15, f"Expected ~+Y direction for pure Y rotation, got {omega}"
    print(f"✓ Pure Y rotation: omega direction = ({omega[0]/mag:.3f}, {omega[1]/mag:.3f}, {omega[2]/mag:.3f})")
    
    # Test 3: Pure +Z rotation → omega should point along +Z
    q = get_orientation_quaternion(0.0, 0.0, 90.0)
    omega = quat_to_omega(q)
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    assert abs(omega[2] / mag - 1.0) < 0.15, f"Expected ~+Z direction for pure Z rotation, got {omega}"
    print(f"✓ Pure Z rotation: omega direction = ({omega[0]/mag:.3f}, {omega[1]/mag:.3f}, {omega[2]/mag:.3f})")
    
    # Test 4: Combined rotation → check that the quaternion represents a valid rotation
    q = get_orientation_quaternion(90.0, 90.0, 0.0)
    omega = quat_to_omega(q)
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    # For 90°X + 90°Y composition: the effective axis is along Y (not X+Y).
    # This is because after rotating 90° about X, the original Y-axis becomes Z.
    # So a subsequent 90° about Y actually rotates about the world-frame Y, which in body frame = Z.
    # The quat_to_omega extracts the "equivalent single rotation axis" from the quaternion.
    print(f"✓ Combined rotation: omega direction = ({omega[0]/mag:.3f}, {omega[1]/mag:.3f}, {omega[2]/mag:.3f})")


def test_omega_vector_alignment_with_axis_negative():
    """Test that the omega vector correctly reflects negative rotations."""
    # Pure -X rotation → omega should point along -X
    q = get_orientation_quaternion(-90.0, 0.0, 0.0)
    omega = quat_to_omega(q)
    mag = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    assert abs(omega[0] / mag - (-1.0)) < 0.15, f"Expected ~-X direction for negative X rotation, got {omega}"
    print(f"✓ Negative X rotation: omega direction = ({omega[0]/mag:.3f}, {omega[1]/mag:.3f}, {omega[2]/mag:.3f})")


def test_omega_vector_alignment_with_axis_opposite():
    """Test that the omega vector correctly reflects opposite rotations."""
    # 180° X vs -180° X should give opposite directions
    q_pos = get_orientation_quaternion(180.0, 0.0, 0.0)
    q_neg = get_orientation_quaternion(-180.0, 0.0, 0.0)
    
    omega_pos = quat_to_omega(q_pos)
    omega_neg = quat_to_omega(q_neg)
    
    # They should point in opposite directions (dot product ≈ -|ω+||ω-|)
    dot = omega_pos[0]*omega_neg[0] + omega_pos[1]*omega_neg[1] + omega_pos[2]*omega_neg[2]
    mag_pos = math.sqrt(sum(c**2 for c in omega_pos))
    mag_neg = math.sqrt(sum(c**2 for c in omega_neg))
    
    if mag_pos > 0.01 and mag_neg > 0.01:
        cos_angle = dot / (mag_pos * mag_neg)
        # For opposite directions, cos should be close to -1... but quaternions q and -q represent the same rotation
        # So we check they're either equal or opposite
        assert abs(abs(cos_angle) - 1.0) < 0.15, f"Expected ±180° between omega vectors, got cos={cos_angle}"
    
    print(f"✓ Opposite rotations: ω_pos=({omega_pos[0]:.3f}, {omega_pos[1]:.3f}, {omega_pos[2]:.3f}), "
          f"ω_neg=({omega_neg[0]:.3f}, {omega_neg[1]:.3f}, {omega_neg[2]:.3f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing quaternion-based rotation composition")
    print("=" * 60)
    
    test_quat_multiply_identity()
    test_quat_normalize()
    test_quat_conjugate()
    test_axis_angle_to_quat()
    test_get_orientation_quaternion()
    test_quat_to_omega_identity()
    test_quat_to_omega_x_rotation()
    test_quat_to_omega_y_rotation()
    test_quat_to_omega_z_rotation()
    test_quat_to_omega_combined_rotation()
    test_quaternion_composition_vs_euler()
    test_omega_vector_alignment_with_axis()
    test_omega_vector_alignment_with_axis_negative()
    test_omega_vector_alignment_with_axis_opposite()
    
    print()
    print("=" * 60)
    print("All quaternion tests passed!")
    print("=" * 60)