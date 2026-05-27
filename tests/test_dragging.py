import pygame
import os
import sys
import pytest
from ctypes import windll, wintypes, WINFUNCTYPE

# Ensure we can import from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from driver import run_simulation
from config import CONFIG

# Centralized screenshot output directory
OUTPUT_DIR = os.path.join(project_root, "screenshots")


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class DragTestState:  # Renamed to avoid pytest collection warning
    def __init__(self):
        self.screenshots = []
        self.angular_velocities = None
        self.drag_started = False
        self.drag_ended = False
        self.first_screenshot_taken = False
        self.second_screenshot_taken = False
        self.frame_count = 0


def maximize_window():
    """Maximize the pygame window by finding it via title and posting WM_SYSCOMMAND SC_MAXIMIZE."""
    try:
        user32 = windll.user32

        ENUM_WINDOWS_PROC = WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM
        )

        found = [False]

        def enum_proc(hwnd, lparam):
            try:
                length = int(user32.GetWindowTextLengthW(hwnd)) + 1
                buf = (wintypes.WCHAR * length)()
                user32.GetWindowTextW(hwnd, buf, length)
                caption = str(buf.value) if hasattr(buf, 'value') else ''
            except Exception:
                return True

            if 'Angular Velocity Demo' in caption:
                WM_SYSCOMMAND = 0x0112
                SC_MAXIMIZE = 0xF030
                user32.PostMessageW(hwnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0)
                found[0] = True
                return False

            return True

        callback = ENUM_WINDOWS_PROC(enum_proc)
        
        old_cb = getattr(user32, '_pygame_drag_test_callback', None)
        user32._pygame_drag_test_callback = callback
        
        user32.EnumWindows(callback, None)
        
        if old_cb is not None:
            user32._pygame_drag_test_callback = old_cb
        else:
            delattr(user32, '_pygame_drag_test_callback')

    except Exception as e:
        print(f"[test_dragging] Window maximize failed (non-fatal): {e}")


def test_cube_dragging():
    state = DragTestState()
    
    # Flag to track that we've already tried maximizing the window
    _maximize_done = [False]

    def screen_callback(screen_surface, fps=0.0, omega=(0.0, 0.0, 0.0)):
        current_time = pygame.time.get_ticks()

        # Try to maximize on first frame (only once) — window must exist by then
        if not _maximize_done[0]:
            try:
                maximize_window()
            except Exception:
                pass
            _maximize_done[0] = True

        # Print FPS and angular velocity every 10 frames
        state.frame_count += 1
        if state.frame_count % 10 == 0:
            print(f"[Frame {state.frame_count}] FPS={fps:.1f}, ω=({omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f})")

        # Screenshot 1: At the very beginning (on first call)
        if not state.first_screenshot_taken:
            path = os.path.join(OUTPUT_DIR, "screenshot_start.png")
            pygame.image.save(screen_surface, path)
            state.screenshots.append(path)
            state.first_screenshot_taken = True

        # Screenshot 2: After drag ends + 1 second (drag ends at ~600ms, so > 1600ms)
        if state.drag_ended and not state.second_screenshot_taken and current_time >= 1600:
            path = os.path.join(OUTPUT_DIR, "screenshot_after_drag.png")
            pygame.image.save(screen_surface, path)
            state.screenshots.append(path)
            state.second_screenshot_taken = True

    def event_injector():
        current_time_ms = pygame.time.get_ticks()
        events = []

        # 1. Start Dragging (at 100ms) - LMB down
        if not state.drag_started and current_time_ms > 100:
            offset_x, offset_y = 50, 50
            click_pos = (CONFIG["screen"]["width"] // 2 + offset_x, CONFIG["screen"]["height"] // 2 + offset_y)
            events.append(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": click_pos, "button": 1}))
            state.drag_started = True

        # 2. Dragging (from 100ms to 600ms) - varying positions to simulate actual dragging
        if state.drag_started and not state.drag_ended and current_time_ms < 600:
            progress = (current_time_ms - 100) / 500.0
            drag_x = int(50 + progress * 150)
            drag_y = int(50 + progress * 120)
            events.append(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (CONFIG["screen"]["width"] // 2 + drag_x, CONFIG["screen"]["height"] // 2 + drag_y), "buttons": (1, 0, 0), "rel": (int(progress * 150), int(progress * 120))}))

        # 3. End Dragging (at 600ms) - LMB up
        if state.drag_started and not state.drag_ended and current_time_ms >= 600:
            events.append(pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": (CONFIG["screen"]["width"] // 2 + 100, CONFIG["screen"]["height"] // 2 + 100), "button": 1}))
            state.drag_ended = True

        return events

    # Run simulation for 3 seconds to allow time for the delay requirement
    omega_x, omega_y, omega_z = run_simulation(duration_seconds=3, event_injector=event_injector, screen_callback=screen_callback)
    state.angular_velocities = (omega_x, omega_y, omega_z)

    # Verification and Output
    print("\n--- Test Results ---")
    print(f"Screenshots taken: {len(state.screenshots)}")
    for s in state.screenshots:
        print(f"  - {s}")
    print(f"Angular Velocities: {state.angular_velocities}")

    assert len(state.screenshots) >= 2, "Should have taken at least two screenshots"
    assert state.angular_velocities[0] != 0, f"Expected non-zero omega_x from drag, got {state.angular_velocities[0]}"
    assert state.angular_velocities[1] != 0, f"Expected non-zero omega_y from drag, got {state.angular_velocities[1]}"