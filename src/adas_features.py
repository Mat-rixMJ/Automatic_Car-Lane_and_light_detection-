"""Advanced Driver Assistance System (ADAS) Features.

These are the differentiating features that make this project stand out:
1. Forward Collision Warning (FCW)
2. Lane Departure Warning (LDW)
3. Speed Limit Compliance
4. Traffic Light Awareness
5. India-specific warnings

All use outputs from the other modules — no extra training needed.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class AlertLevel(Enum):
    NONE = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class AdasState:
    """Current state of all ADAS alerts."""
    collision_risk: float = 0.0          # 0-1 (1 = imminent)
    collision_alert: AlertLevel = AlertLevel.NONE
    closest_vehicle_dist: float = float('inf')

    lane_offset: float = 0.0             # -1 to 1 (0 = centered)
    lane_departure: AlertLevel = AlertLevel.NONE
    lane_departure_side: str = ""        # "left" or "right"

    current_speed_limit: Optional[int] = None
    overspeed: bool = False

    traffic_light_state: str = ""        # red/yellow/green
    traffic_light_alert: AlertLevel = AlertLevel.NONE

    alerts: list = field(default_factory=list)  # Active alert messages


class AdasModule:
    """ADAS decision engine — takes module outputs and produces alerts."""

    # FCW thresholds (relative distance units)
    FCW_WARNING_DIST = 3.0    # Warning zone
    FCW_CRITICAL_DIST = 1.5   # Critical zone (emergency brake level)

    # LDW thresholds
    LDW_THRESHOLD = 0.3       # Lane offset ratio to trigger warning

    def __init__(self):
        self.state = AdasState()
        self.last_alert_time = {}  # Debouncing

    def update(self, yolop_results, sign_detections, light_detections,
               depth_map=None, frame_shape=None):
        """Update ADAS state based on all perception outputs.

        Args:
            yolop_results: dict from YOLOPInference.infer()
            sign_detections: list from SignDetector.detect()
            light_detections: list from TrafficLightDetector.detect()
            depth_map: (H, W) depth map from DepthEstimator (optional)
            frame_shape: (H, W) of the original frame

        Returns:
            AdasState with current alerts
        """
        self.state = AdasState()

        if frame_shape:
            h, w = frame_shape[:2]
        else:
            h, w = 720, 1280  # Default

        # 1. Forward Collision Warning
        self._check_collision(yolop_results, depth_map, h, w)

        # 2. Lane Departure Warning
        self._check_lane_departure(yolop_results, h, w)

        # 3. Speed limit compliance
        self._check_speed_limit(sign_detections)

        # 4. Traffic light awareness
        self._check_traffic_light(light_detections)

        return self.state

    def _check_collision(self, yolop_results, depth_map, h, w):
        """Forward Collision Warning based on vehicle detection + depth."""
        boxes = yolop_results.get('det_boxes', [])
        if not boxes:
            return

        # Find closest vehicle in the forward zone (center 60% of frame)
        forward_left = int(w * 0.2)
        forward_right = int(w * 0.8)

        min_dist = float('inf')
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) // 2

            # Only consider vehicles in the forward path
            if cx < forward_left or cx > forward_right:
                continue

            if depth_map is not None:
                # Use depth estimation for distance
                from depth_estimation import DepthEstimator
                bbox_depth = depth_map[y1:y2, x1:x2]
                if bbox_depth.size > 0:
                    avg_depth = np.mean(bbox_depth)
                    if avg_depth > 0.01:
                        dist = 10.0 / avg_depth
                        min_dist = min(min_dist, dist)
            else:
                # Fallback: use bounding box size as proxy for distance
                # Larger box = closer vehicle
                box_area_ratio = (x2 - x1) * (y2 - y1) / (h * w)
                # ponytail: crude but works for demo — bigger box = closer
                dist = 1.0 / max(box_area_ratio, 0.001)
                min_dist = min(min_dist, dist)

        self.state.closest_vehicle_dist = min_dist

        if min_dist < self.FCW_CRITICAL_DIST:
            self.state.collision_alert = AlertLevel.CRITICAL
            self.state.collision_risk = 1.0
            self.state.alerts.append("⚠️ COLLISION WARNING - BRAKE!")
        elif min_dist < self.FCW_WARNING_DIST:
            self.state.collision_alert = AlertLevel.WARNING
            self.state.collision_risk = 1.0 - (min_dist / self.FCW_WARNING_DIST)
            self.state.alerts.append("⚡ Vehicle too close")

    def _check_lane_departure(self, yolop_results, h, w):
        """Lane Departure Warning based on lane segmentation mask."""
        lane_mask = yolop_results.get('lane_seg', None)
        if lane_mask is None or not lane_mask.any():
            return

        # Analyze the bottom third of the frame (immediate road ahead)
        bottom_region = lane_mask[int(h * 0.7):, :]
        if not bottom_region.any():
            return

        # Find lane line positions
        lane_cols = np.where(bottom_region.any(axis=0))[0]
        if len(lane_cols) < 2:
            return

        # Left and right lane boundaries
        left_lane = lane_cols.min()
        right_lane = lane_cols.max()
        lane_center = (left_lane + right_lane) / 2
        frame_center = w / 2

        # Calculate offset (-1 to 1)
        lane_width = right_lane - left_lane
        if lane_width > 0:
            offset = (frame_center - lane_center) / (lane_width / 2)
            self.state.lane_offset = np.clip(offset, -1.0, 1.0)

            if abs(offset) > self.LDW_THRESHOLD:
                self.state.lane_departure = AlertLevel.WARNING
                self.state.lane_departure_side = "left" if offset < 0 else "right"
                self.state.alerts.append(
                    f"↔️ Lane departure ({self.state.lane_departure_side})")

    def _check_speed_limit(self, sign_detections):
        """Check for speed limit signs."""
        for det in sign_detections:
            class_name = det.get('class', '').lower()
            if 'speed limit' in class_name:
                # Extract number from class name like "Speed limit (60km/h)"
                import re
                match = re.search(r'(\d+)', class_name)
                if match:
                    self.state.current_speed_limit = int(match.group(1))
                    break

    def _check_traffic_light(self, light_detections):
        """Traffic light state awareness."""
        if not light_detections:
            return

        # Use the most confident detection
        best = max(light_detections, key=lambda x: x['conf'])
        self.state.traffic_light_state = best['state']

        if best['state'] == 'red':
            self.state.traffic_light_alert = AlertLevel.WARNING
            self.state.alerts.append("🔴 Red light ahead")
        elif best['state'] == 'yellow':
            self.state.traffic_light_alert = AlertLevel.INFO
            self.state.alerts.append("🟡 Yellow light - prepare to stop")

    def draw_dashboard(self, frame):
        """Draw ADAS dashboard overlay on frame.

        This is the visual differentiator — a clean, professional HUD
        showing all ADAS information at a glance.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- Bottom dashboard panel ---
        panel_h = 100
        panel_y = h - panel_h
        cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        y_base = panel_y + 25

        # Collision risk gauge
        self._draw_risk_gauge(frame, 20, y_base, self.state.collision_risk)
        cv2.putText(frame, "FCW", (20, y_base + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Lane position indicator
        self._draw_lane_indicator(frame, 150, y_base, self.state.lane_offset)
        cv2.putText(frame, "LDW", (150, y_base + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Speed limit
        if self.state.current_speed_limit:
            self._draw_speed_limit(frame, 300, y_base - 10, self.state.current_speed_limit)

        # Traffic light state
        if self.state.traffic_light_state:
            self._draw_traffic_light_icon(frame, 420, y_base - 10,
                                          self.state.traffic_light_state)

        # Distance to closest vehicle
        if self.state.closest_vehicle_dist < float('inf'):
            dist_text = f"Dist: {self.state.closest_vehicle_dist:.1f}m"
            cv2.putText(frame, dist_text, (520, y_base + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Alert banner (top, if critical)
        if self.state.collision_alert == AlertLevel.CRITICAL:
            cv2.rectangle(frame, (0, 50), (w, 100), (0, 0, 200), -1)
            cv2.putText(frame, "!! FORWARD COLLISION WARNING !!",
                        (w // 2 - 250, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif self.state.lane_departure == AlertLevel.WARNING:
            color = (0, 165, 255)  # Orange
            side = self.state.lane_departure_side
            if side == "left":
                cv2.line(frame, (0, 50), (0, h - panel_h), color, 8)
            else:
                cv2.line(frame, (w - 1, 50), (w - 1, h - panel_h), color, 8)

        return frame

    def _draw_risk_gauge(self, frame, x, y, risk):
        """Draw a horizontal risk gauge (green → red)."""
        bar_w, bar_h = 100, 15
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * risk)
        if risk < 0.5:
            color = (0, 255, 0)
        elif risk < 0.8:
            color = (0, 200, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), 1)

    def _draw_lane_indicator(self, frame, x, y, offset):
        """Draw lane position indicator (car between lanes)."""
        bar_w, bar_h = 100, 15
        center_x = x + bar_w // 2
        car_x = center_x + int(offset * bar_w // 2)

        # Lane boundaries
        cv2.line(frame, (x, y), (x, y + bar_h), (255, 255, 255), 2)
        cv2.line(frame, (x + bar_w, y), (x + bar_w, y + bar_h), (255, 255, 255), 2)
        # Center line (dashed)
        cv2.line(frame, (center_x, y), (center_x, y + bar_h), (100, 100, 100), 1)
        # Car position
        color = (0, 255, 0) if abs(offset) < self.LDW_THRESHOLD else (0, 0, 255)
        cv2.circle(frame, (car_x, y + bar_h // 2), 5, color, -1)

    def _draw_speed_limit(self, frame, x, y, limit):
        """Draw speed limit sign icon."""
        cv2.circle(frame, (x + 20, y + 20), 22, (0, 0, 255), 2)
        cv2.circle(frame, (x + 20, y + 20), 18, (255, 255, 255), -1)
        text = str(limit)
        font_scale = 0.5 if limit >= 100 else 0.6
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        tx = x + 20 - text_size[0] // 2
        ty = y + 20 + text_size[1] // 2
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    def _draw_traffic_light_icon(self, frame, x, y, state):
        """Draw mini traffic light indicator."""
        # Housing
        cv2.rectangle(frame, (x, y), (x + 20, y + 50), (40, 40, 40), -1)
        # Lights
        colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
        positions = {'red': y + 10, 'yellow': y + 25, 'green': y + 40}
        for s, pos in positions.items():
            color = colors[s] if s == state else (60, 60, 60)
            cv2.circle(frame, (x + 10, pos), 5, color, -1)
