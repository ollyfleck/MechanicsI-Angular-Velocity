import os
import time
import sys

# Ensure we can import from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pygame  # Import inside to be sure it's in the module scope
from driver import run_simulation

# Centralized screenshot output directory
SCREENSHOTS_DIR = os.path.join(project_root, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

class DragEventInjector:
    def __init__(self):
        # We will create events on the fly to avoid any scope issues with pre-created objects
        self.index = 0
        self.events_data = [
            ('MOUSEBUTTONDOWN', {'button': 1, 'pos': (400, 300)}),
            ('MOUSEMOTION', {'button': 1, 'pos': (500, 400)}),
            ('MOUSEBUTTONUP', {'button': 1, 'pos': (500, 400)})
        ]

    def __call__(self):
        import pygame  # Import inside to be absolutely sure
        if self.index < len(self.events_data):
            event_type_str, event_args = self.events_data[self.index]
            # Map the string name back to the pygame constant
            event_type = getattr(pygame, event_type_str)
            event = pygame.event.Event(event_type, event_args)
            self.index += 1
            return [event]
        return []

def test_drag_interaction():
    print("Starting integration test: Dragging the cube...")
    
    screenshots = []
    start_time = time.time()

    def screen_callback(surface):
        elapsed = time.time() - start_time
        # Capture screenshot at start and after 1 second (approx)
        if len(screenshots) == 0 or (elapsed > 1.0 and len(screenshots) < 2):
            filename = f"test_screenshot_{len(screenshots)}.png"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            pygame.image.save(surface, filepath)
            screenshots.append(filepath)
            print(f"Captured screenshot: {filepath}")

    injector = DragEventInjector()
    
    # Run simulation for 2 seconds to ensure we cover the drag and the 1s wait
    try:
        omega_x, omega_y, omega_z = run_simulation(duration_seconds=2.0, event_injector=injector, screen_callback=screen_callback)
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        raise e

    print(f"Final angular velocity: omega_x={omega_x:.4f}, omega_y={omega_y:.4f}, omega_z={omega_z:.4f}")

    # Check if we actually got some rotation
    total_mag = (omega_x**2 + omega_y**2 + omega_z**2)**0.5
    print(f"Total angular velocity magnitude: {total_mag:.4f}")

    assert total_mag > 0.01, f"Drag failed to impart momentum! Omega was: ({omega_x:.4f}, {omega_y:.4f}, {omega_z:.4f})"
    assert len(screenshots) >= 1, "No screenshots were captured!"
    print("Integration test passed!")

if __name__ == "__main__":
    try:
        test_drag_interaction()
        print("SUCCESS")
    except Exception as e:
        print(f"FAILURE: {e}")
        sys.exit(1)