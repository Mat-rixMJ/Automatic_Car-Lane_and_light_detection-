"""CARLA Vehicle Controller — Perception-driven driving.

Takes ADAS decisions (from adas_features.py) and translates them into
actual vehicle control commands in CARLA:
- Emergency brake on collision warning
- Gentle brake on red traffic light
- Throttle reduction on yellow light
- Lane correction on departure warning
- Speed limit enforcement

This closes the loop: Perception → Decision → Action.
"""

import time
import math

try:
    import carla
except ImportError:
    carla = None


class CarlaController:
    """Translates ADAS alerts into CARLA vehicle control commands.

    Control flow:
        1. Perception pipeline runs on camera frame
        2. ADAS module produces alerts (collision, lane departure, red light, etc.)
        3. This controller converts alerts into VehicleControl (brake/throttle/steer)
        4. Control is applied to the CARLA vehicle
    """

    def __init__(self, vehicle, default_speed_kmh=40):
        """
        Args:
            vehicle: carla.Vehicle actor
            default_speed_kmh: Target cruising speed when no alerts
        """
        self.vehicle = vehicle
        self.default_speed = default_speed_kmh
        self.target_speed = default_speed_kmh
        self.is_braking = False
        self.brake_start_time = 0

        # PID-like parameters for smooth control
        self.throttle_kp = 0.5
        self.steer_correction_kp = 0.3

        # State
        self.autopilot_active = False

    def set_manual_control(self):
        """Disable CARLA autopilot — our pipeline controls the car."""
        self.vehicle.set_autopilot(False)
        self.autopilot_active = False
        print("Vehicle control: MANUAL (perception-driven)")

    def set_autopilot(self):
        """Re-enable CARLA autopilot (hands off)."""
        self.vehicle.set_autopilot(True)
        self.autopilot_active = True
        print("Vehicle control: AUTOPILOT")

    def get_current_speed_kmh(self):
        """Get current vehicle speed in km/h."""
        vel = self.vehicle.get_velocity()
        speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed_ms * 3.6  # m/s to km/h

    def compute_control(self, adas_state):
        """Compute vehicle control from ADAS state.

        Args:
            adas_state: AdasState from adas_features.py

        Returns:
            carla.VehicleControl
        """
        if carla is None:
            return None

        from adas_features import AlertLevel

        control = carla.VehicleControl()
        current_speed = self.get_current_speed_kmh()

        # --- Priority 1: Emergency brake (collision) ---
        if adas_state.collision_alert == AlertLevel.CRITICAL:
            control.brake = 1.0
            control.throttle = 0.0
            self.is_braking = True
            self.brake_start_time = time.time()
            return control

        # --- Priority 2: Brake for red light ---
        if adas_state.traffic_light_state == "red":
            control.brake = 0.6
            control.throttle = 0.0
            self.is_braking = True
            return control

        # --- Priority 3: Slow down for yellow light ---
        if adas_state.traffic_light_state == "yellow":
            control.brake = 0.3
            control.throttle = 0.0
            return control

        # --- Priority 4: Collision warning (not critical) — reduce speed ---
        if adas_state.collision_alert == AlertLevel.WARNING:
            control.brake = 0.4
            control.throttle = 0.0
            return control

        # --- Priority 5: Speed limit enforcement ---
        if adas_state.current_speed_limit:
            self.target_speed = adas_state.current_speed_limit
        else:
            self.target_speed = self.default_speed

        # --- Priority 6: Lane departure correction ---
        steer = 0.0
        if adas_state.lane_departure == AlertLevel.WARNING:
            # Steer back toward center
            offset = adas_state.lane_offset
            steer = -offset * self.steer_correction_kp
            steer = max(-0.3, min(0.3, steer))  # Gentle correction only

        control.steer = steer

        # --- Normal driving: maintain target speed ---
        speed_error = self.target_speed - current_speed

        if speed_error > 2:
            # Need to speed up
            control.throttle = min(0.7, speed_error * self.throttle_kp * 0.05)
            control.brake = 0.0
        elif speed_error < -5:
            # Over speed limit — brake gently
            control.throttle = 0.0
            control.brake = min(0.3, abs(speed_error) * 0.02)
        else:
            # Cruising — maintain
            control.throttle = 0.3
            control.brake = 0.0

        self.is_braking = False
        return control

    def apply_control(self, adas_state):
        """Compute and apply control to the CARLA vehicle.

        Args:
            adas_state: AdasState from ADAS module

        Returns:
            carla.VehicleControl that was applied (for logging/display)
        """
        if self.autopilot_active:
            return None

        control = self.compute_control(adas_state)
        if control:
            self.vehicle.apply_control(control)
        return control

    def emergency_stop(self):
        """Immediately stop the vehicle."""
        if carla is None:
            return
        control = carla.VehicleControl()
        control.brake = 1.0
        control.throttle = 0.0
        control.hand_brake = True
        self.vehicle.apply_control(control)
        self.is_braking = True
        print("EMERGENCY STOP applied")

    def get_status_text(self, control):
        """Get human-readable control status for HUD display."""
        if control is None:
            return "AUTOPILOT"

        speed = self.get_current_speed_kmh()
        parts = [f"Speed: {speed:.0f} km/h"]

        if control.brake > 0.5:
            parts.append("BRAKING")
        elif control.brake > 0:
            parts.append(f"Brake: {control.brake:.0%}")

        if control.throttle > 0:
            parts.append(f"Throttle: {control.throttle:.0%}")

        if abs(control.steer) > 0.01:
            direction = "L" if control.steer < 0 else "R"
            parts.append(f"Steer: {direction} {abs(control.steer):.2f}")

        parts.append(f"Target: {self.target_speed:.0f} km/h")

        return " | ".join(parts)
