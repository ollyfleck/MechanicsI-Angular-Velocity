"""
FPS benchmark test - measures framerate at different angular velocity levels.
Tests for performance degradation when vector length is below tip_length.
"""

import pygame
import time
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import CONFIG, SCREEN_W, SCREEN_H
from draw_3d import (
    compute_vector_alpha,
    draw_3d_cylinder,
    draw_3d_cone,
    draw_3d_vector_omega_drawables,
    compute_dynamic_segments,
)
from projection import project_3d_to_screen


def setup_pygame():
    """Initialize pygame for surface creation with minimized window."""
    if not pygame.get_init():
        # Minimize the window so it doesn't interfere
        os.environ['SDL_VIDEO_WINDOW_POS'] = '-1000,-1000'
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        # Minimize window
        pygame.display.set_icon(pygame.Surface((1, 1)))
        return screen
    return pygame.display.get_surface()


def measure_fps(draw_func, *args, num_frames=60):
    """Measure FPS for a draw function by running it many times."""
    screen = setup_pygame()
    
    # Warm up
    for _ in range(10):
        draw_func(*args)
    
    # Time the frames
    start = time.perf_counter()
    for _ in range(num_frames):
        draw_func(*args)
    elapsed = time.perf_counter() - start
    
    return num_frames / elapsed if elapsed > 0 else 9999


def test_omega_drawables_low_velocity():
    """Test FPS for omega vector drawables at very low angular velocity (below tip_length)."""
    screen = setup_pygame()
    
    # Get config values
    geom_config = CONFIG.get('vectors', {}).get('omega', {}).get('geometry', {})
    tip_length = geom_config.get('tip_length', 18.0)
    shaft_segments = geom_config.get('shaft_segments', 8)
    
    # Very low velocity: total_length = 0.001 * 0.5 + 20 = 20.0005 (just barely above base)
    # This is well below tip_length (18.0) for the visible portion
    low_omega = 0.001
    direction = (1.0, 0.0, 0.0)
    center = (0.0, 0.0, 0.0)
    max_speed = CONFIG['angular_velocity']['max_speed']
    
    # Create a callable that mimics the draw call
    def draw_low():
        result, _ = draw_3d_vector_omega_drawables(
            center, direction, low_omega, {}, geom_config, max_speed, screen
        )
        return result
    
    fps = measure_fps(draw_low, num_frames=1000)
    
    # At low velocity with alpha vectors, should still be fast (>100 FPS)
    print(f"Low velocity (omega={low_omega}, below tip_length): {fps:.1f} FPS")
    
    # Count how many drawables are created
    drawables, _ = draw_3d_vector_omega_drawables(
        center, direction, low_omega, {}, geom_config, max_speed, screen
    )
    num_drawables = len(drawables) if drawables else 0
    print(f"  Drawables created: {num_drawables}")
    
    return fps, num_drawables


def test_omega_drawables_high_velocity():
    """Test FPS for omega vector drawables at high angular velocity (above tip_length)."""
    screen = setup_pygame()
    
    geom_config = CONFIG.get('vectors', {}).get('omega', {}).get('geometry', {})
    max_speed = CONFIG['angular_velocity']['max_speed']
    
    high_omega = 5.0
    direction = (1.0, 0.0, 0.0)
    center = (0.0, 0.0, 0.0)
    
    def draw_high():
        result, _ = draw_3d_vector_omega_drawables(
            center, direction, high_omega, {}, geom_config, max_speed, screen
        )
        return result
    
    fps = measure_fps(draw_high, num_frames=1000)
    
    print(f"High velocity (omega={high_omega}, above tip_length): {fps:.1f} FPS")
    
    drawables, _ = draw_3d_vector_omega_drawables(
        center, direction, high_omega, {}, geom_config, max_speed, screen
    )
    num_drawables = len(drawables) if drawables else 0
    print(f"  Drawables created: {num_drawables}")
    
    return fps, num_drawables


def test_omega_drawables_zero_velocity():
    """Test FPS for omega vector drawables at zero angular velocity."""
    screen = setup_pygame()
    
    geom_config = CONFIG.get('vectors', {}).get('omega', {}).get('geometry', {})
    max_speed = CONFIG['angular_velocity']['max_speed']
    
    zero_omega = 0.0
    direction = (1.0, 0.0, 0.0)
    center = (0.0, 0.0, 0.0)
    
    def draw_zero():
        result, _ = draw_3d_vector_omega_drawables(
            center, direction, zero_omega, {}, geom_config, max_speed, screen
        )
        return result
    
    fps = measure_fps(draw_zero, num_frames=1000)
    
    print(f"Zero velocity (omega={zero_omega}): {fps:.1f} FPS")
    
    drawables, _ = draw_3d_vector_omega_drawables(
        center, direction, zero_omega, {}, geom_config, max_speed, screen
    )
    num_drawables = len(drawables) if drawables else 0
    print(f"  Drawables created: {num_drawables}")
    
    return fps, num_drawables


def test_overlay_surface_creation():
    """Test FPS impact of overlay surface creation."""
    screen = setup_pygame()
    
    # Test creating overlay surfaces (this is what happens in the old code)
    def create_overlay():
        alpha = 50
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.set_alpha(alpha)
        return overlay
    
    fps = measure_fps(create_overlay, num_frames=1000)
    
    print(f"Overlay surface creation: {fps:.1f} FPS")
    
    # Test filling the overlay with a polygon (also in the old code)
    def create_and_fill_overlay():
        alpha = 50
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.set_alpha(alpha)
        pygame.draw.polygon(overlay, (100, 100, 255), [(100, 100), (200, 100), (150, 200)], 0)
        return overlay
    
    fps_fill = measure_fps(create_and_fill_overlay, num_frames=1000)
    
    print(f"Overlay + polygon fill: {fps_fill:.1f} FPS")
    
    return fps, fps_fill


def test_dynamic_segments():
    """Test that dynamic segment scaling is working correctly."""
    # Test with a short vector (should return 4 segments)
    start_3d = (0.0, 0.0, 0.0)
    end_3d = (0.001, 0.0, 0.0)  # Very short
    segs, length = compute_dynamic_segments(8, start_3d, end_3d)
    print(f"Short vector: {segs} segments, projected length: {length:.1f}px")
    
    # Test with a longer vector (should return 8 segments)
    end_3d = (10.0, 0.0, 0.0)  # Longer
    segs, length = compute_dynamic_segments(8, start_3d, end_3d)
    print(f"Long vector: {segs} segments, projected length: {length:.1f}px")


def run_all_tests():
    """Run all FPS benchmark tests."""
    print("=" * 60)
    print("FPS BENCHMARK TESTS")
    print("=" * 60)
    print()
    
    # Setup pygame with hidden window
    screen = setup_pygame()
    print(f"Screen size: {SCREEN_W}x{SCREEN_H}")
    print("(Window is hidden off-screen)")
    print()
    
    # Test dynamic segments
    print("--- Dynamic Segment Scaling ---")
    test_dynamic_segments()
    print()
    
    # Test omega at different velocities
    print("--- Omega Vector FPS ---")
    fps_low, drawables_low = test_omega_drawables_low_velocity()
    fps_high, drawables_high = test_omega_drawables_high_velocity()
    fps_zero, drawables_zero = test_omega_drawables_zero_velocity()
    print()
    
    # Test overlay surface creation
    print("--- Overlay Surface Creation ---")
    fps_overlay, fps_overlay_fill = test_overlay_surface_creation()
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    issues = []
    
    if fps_low < 100:
        issues.append(f"LOW FPS at low velocity: {fps_low:.1f} FPS (expected >100)")
    
    if fps_zero < 100:
        issues.append(f"LOW FPS at zero velocity: {fps_zero:.1f} FPS (expected >100)")
    
    if drawables_low > 0:
        issues.append(f"Drawables created at LOW velocity: {drawables_low} (expected 0)")
    
    if drawables_zero > 0:
        issues.append(f"Drawables created at ZERO velocity: {drawables_zero} (expected 0)")
    
    if fps_overlay < 500:
        issues.append(f"Overlay creation is slow: {fps_overlay:.1f} FPS")
    
    if fps_overlay_fill < 200:
        issues.append(f"Overlay + fill is slow: {fps_overlay_fill:.1f} FPS")
    
    if drawables_high > drawables_low * 3 and drawables_low > 0:
        issues.append(f"High velocity doesn't scale well: {drawables_high} vs {drawables_low}")
    
    if issues:
        print("POTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major performance issues detected.")
    
    print()
    print("NOTE: If overlay FPS is low (<500), the overlay surface creation")
    print("is likely the bottleneck when vectors are below tip_length.")
    
    return issues


if __name__ == "__main__":
    try:
        issues = run_all_tests()
        if issues:
            print(f"\n{len(issues)} issue(s) found. See details above.")
            sys.exit(1)
        else:
            print("\nAll tests passed - no major issues found.")
            sys.exit(0)
    finally:
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()
