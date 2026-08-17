"""CARLA Simulator Bridge — Perception + Control Loop.

Connects to a running CARLA server, spawns a vehicle with a front-facing camera,
runs the perception pipeline, and optionally controls the vehicle based on
ADAS decisions (brake on collision, stop on red light, lane correction).

Modes:
    --mode observe   : Autopilot drives, pipeline observes and annotates only
    --mode control   : Pipeline controls the vehicle (brakes, steers, throttle)

Requirements:
    - CARLA simulator running (0.9.13+ recommended)
    - carla Python package: pip install carla

Usage:
    1. Start CARLA: CarlaUE4.exe
    2. Observe mode: python src/carla_bridge.py --mode observe
    3. Control mode: python src/carla_bridge.py --mode control
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import carla
except ImportError:
    carla = None


class CarlaBridge:
    """Bridge between CARLA simulator and the perception pipeline."""

    def __init__(self, host="localhost", port=2000, resolution=(800, 600)):
        if carla is None:
            print("ERROR: 'carla' package not found.")
            print("Install with: pip install carla")
            sys.exit(1)

        self.host = host
        self.port = port
        self.width, self.height = resolution
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.latest_frame = None

    def connect(self):
        """Connect to CARLA server."""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print(f"Connected. Map: {self.world.get_map().name}")

    def spawn_vehicle(self):
        """Spawn a vehicle at a random spawn point (no autopilot by default)."""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned: {self.vehicle.type_id}")

    def attach_camera(self):
        """Attach front-facing RGB camera."""
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.width))
        cam_bp.set_attribute("image_size_y", str(self.height))
        cam_bp.set_attribute("fov", "90")

        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5)
        )
        self.camera = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
        self.camera.listen(self._on_frame)
        print(f"Camera attached ({self.width}x{self.height})")

    def _on_frame(self, image):
        """Camera callback: BGRA → BGR numpy array."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((self.height, self.width, 4))
        self.latest_frame = arr[:, :, :3]

    def run_observe_mode(self, output_path=None):
        """Observe mode: CARLA autopilot drives, pipeline annotates.

        Use this to demo the perception without touching vehicle control.
        """
        self.vehicle.set_autopilot(True)
        print("Mode: OBSERVE (autopilot drives, pipeline watches)")
        self._run_loop(output_path, control_enabled=False)

    def run_control_mode(self, output_path=None, default_speed=40):
        """Control mode: Perception pipeline drives the vehicle.

        The vehicle is controlled entirely by ADAS decisions:
        - Red light → brake
        - Collision risk → emergency brake
        - Lane departure → steer correction
        - Speed limit sign → speed adjustment
        - All clear → cruise at target speed
        """
        from carla_controller import CarlaController

        self.controller = CarlaController(self.vehicle, default_speed_kmh=default_speed)
        self.controller.set_manual_control()
        print(f"Mode: CONTROL (perception drives the car, target {default_speed} km/h)")
        self._run_loop(output_path, control_enabled=True)

    def _run_loop(self, output_path, control_enabled):
        """Main perception + control loop."""
        from pipeline import PerceptionPipeline

        pipeline = PerceptionPipeline()

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 20, (self.width, self.height))

        print("\nRunning... Press 'q' to quit, 'b' for emergency brake, 'a' to toggle autopilot\n")

        frame_count = 0
        start_time = time.time()
        last_control = None

        while True:
            frame = self.latest_frame
            if frame is None:
                time.sleep(0.01)
                continue

            # --- Perception ---
            result = pipeline.process_frame(frame)
            frame_count += 1

            # --- Control (if enabled) ---
            if control_enabled and hasattr(self, 'controller'):
                adas_state = pipeline.adas.state
                last_control = self.controller.apply_control(adas_state)

                # Draw control status on frame
                status = self.controller.get_status_text(last_control)
                self._draw_control_hud(result, status, last_control)

            # --- FPS ---
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            mode_text = "CONTROL" if control_enabled else "OBSERVE"
            cv2.putText(result, f"CARLA [{mode_text}] FPS: {fps:.1f}",
                        (self.width - 280, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if writer:
                writer.write(result)

            cv2.imshow("CarLaneI - CARLA", result)

            # --- Keyboard input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b') and control_enabled:
                # Manual emergency brake
                self.controller.emergency_stop()
                print("Manual emergency brake!")
            elif key == ord('a') and control_enabled:
                # Toggle autopilot
                if self.controller.autopilot_active:
                    self.controller.set_manual_control()
                else:
                    self.controller.set_autopilot()

        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nDone: {frame_count} frames, {fps:.1f} FPS avg")

    def _draw_control_hud(self, frame, status_text, control):
        """Draw vehicle control information on frame."""
        h, w = frame.shape[:2]
        y = h - 120  # Above the ADAS dashboard

        # Control status bar
        cv2.putText(frame, status_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if control is None:
            return

        # Visual brake/throttle indicators
        bar_x = w - 60
        bar_h = 80
        bar_y = h - 200

        # Throttle bar (green, goes up)
        throttle_h = int(bar_h * control.throttle)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y + bar_h - throttle_h),
                      (bar_x + 20, bar_y + bar_h), (0, 200, 0), -1)
        cv2.putText(frame, "T", (bar_x + 5, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        # Brake bar (red, goes up)
        bar_x2 = w - 30
        brake_h = int(bar_h * control.brake)
        cv2.rectangle(frame, (bar_x2, bar_y), (bar_x2 + 20, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x2, bar_y + bar_h - brake_h),
                      (bar_x2 + 20, bar_y + bar_h), (0, 0, 200), -1)
        cv2.putText(frame, "B", (bar_x2 + 5, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

        # Steering indicator
        steer_cx = w - 45
        steer_cy = bar_y + bar_h + 25
        steer_len = int(30 * control.steer)
        cv2.line(frame, (steer_cx, steer_cy), (steer_cx + steer_len, steer_cy),
                 (255, 255, 0), 3)
        cv2.circle(frame, (steer_cx, steer_cy), 3, (255, 255, 255), -1)

    def cleanup(self):
        """Destroy spawned actors."""
        # Stop vehicle
        if self.vehicle:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            time.sleep(0.1)
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("Cleaned up CARLA actors")


# --- CLI ---
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import PROJECT_ROOT, ensure_dirs

    parser = argparse.ArgumentParser(description="CarLaneI + CARLA Simulator")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--width", type=int, default=800, help="Camera width")
    parser.add_argument("--height", type=int, default=600, help="Camera height")
    parser.add_argument("--mode", default="control", choices=["observe", "control"],
                        help="observe: autopilot + perception | control: perception drives")
    parser.add_argument("--speed", type=int, default=40, help="Target speed in km/h")
    parser.add_argument("--output", type=str, help="Output video path")
    args = parser.parse_args()

    ensure_dirs()
    bridge = CarlaBridge(args.host, args.port, (args.width, args.height))

    try:
        bridge.connect()
        bridge.spawn_vehicle()
        bridge.attach_camera()
        time.sleep(1.0)  # Let CARLA produce initial frames

        if args.mode == "observe":
            bridge.run_observe_mode(output_path=args.output)
        else:
            bridge.run_control_mode(output_path=args.output, default_speed=args.speed)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        bridge.cleanup()
