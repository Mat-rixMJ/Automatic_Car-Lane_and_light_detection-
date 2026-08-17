"""Traffic Light Detection and State Classification.

Uses YOLOv8-nano to detect traffic lights and classify their state
(red, yellow, green). Falls back to color-based HSV detection.

Datasets for training:
- LISA Traffic Light: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset
- Bosch Small Traffic Lights: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
- DTLD (DriveU): https://www.uni-ulm.de/en/in/driveu/projects/driveu-traffic-light-dataset/
"""

import sys
from pathlib import Path

import cv2
import numpy as np


class TrafficLightDetector:
    """Detects traffic lights and determines their state."""

    STATES = ["red", "yellow", "green"]
    STATE_COLORS = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "unknown": (128, 128, 128)
    }

    def __init__(self, model_path=None, conf_threshold=0.5, use_coco_model=None):
        self.conf_threshold = conf_threshold
        self.detector = None
        self.use_coco = False

        # Strategy: dedicated traffic light model > YOLOv8n COCO fallback > color detection
        if model_path and Path(model_path).exists():
            from ultralytics import YOLO
            self.detector = YOLO(model_path)
            print(f"Traffic light detector loaded: {model_path}")
        elif use_coco_model and Path(use_coco_model).exists():
            from ultralytics import YOLO
            self.detector = YOLO(use_coco_model)
            self.use_coco = True
            # ponytail: COCO class 9 = traffic light. No training needed.
            print(f"Traffic light: Using YOLOv8 COCO (class 9 = traffic light)")
        else:
            print("Traffic light: Using color-based detection (no YOLOv8 model)")

    def detect(self, frame):
        """Detect traffic lights and their state.

        Args:
            frame: BGR image

        Returns:
            list of dicts: [{'bbox': [x1,y1,x2,y2], 'state': str, 'conf': float}]
        """
        if self.detector:
            return self._detect_yolo(frame)
        else:
            return self._detect_color(frame)

    def _detect_yolo(self, frame):
        """Detect using YOLOv8 (dedicated or COCO model)."""
        results = self.detector(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])

                # If using COCO model, only keep class 9 (traffic light)
                if self.use_coco and cls_id != 9:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                if self.use_coco:
                    # COCO model detects the light but not its state
                    # Determine state by analyzing color in the ROI
                    roi = frame[y1:y2, x1:x2]
                    state = self._classify_light_state(roi)
                else:
                    # Dedicated model outputs state directly
                    state = self.STATES[cls_id] if cls_id < len(self.STATES) else "unknown"

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'state': state,
                    'conf': conf
                })

        return detections

    def _classify_light_state(self, roi):
        """Determine traffic light state from a cropped ROI using color analysis."""
        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, w = roi.shape[:2]

        # Split ROI into thirds (top=red, middle=yellow, bottom=green)
        third = h // 3
        top = hsv[:third, :]
        mid = hsv[third:2*third, :]
        bot = hsv[2*third:, :]

        # Check brightness (V channel) in each region
        top_brightness = np.mean(top[:, :, 2])
        mid_brightness = np.mean(mid[:, :, 2])
        bot_brightness = np.mean(bot[:, :, 2])

        brightest = max(top_brightness, mid_brightness, bot_brightness)
        if brightest < 80:
            return "unknown"

        if top_brightness == brightest:
            return "red"
        elif mid_brightness == brightest:
            return "yellow"
        else:
            return "green"

    def _detect_color(self, frame):
        """Fallback: detect traffic lights using color analysis.

        Strategy:
        1. Look for dark rectangular regions in upper half of frame (light housing)
        2. Check for bright red/yellow/green circles inside
        """
        h, w = frame.shape[:2]
        # Traffic lights are typically in the upper 60% of the frame
        roi = frame[:int(h * 0.6), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        detections = []

        # Detect each light color separately
        color_ranges = {
            "red": [
                (np.array([0, 150, 150]), np.array([10, 255, 255])),
                (np.array([170, 150, 150]), np.array([180, 255, 255]))
            ],
            "yellow": [
                (np.array([20, 150, 150]), np.array([35, 255, 255]))
            ],
            "green": [
                (np.array([40, 100, 100]), np.array([90, 255, 255]))
            ]
        }

        for state, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # Look for circular blobs
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            circles = cv2.HoughCircles(
                mask, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=15, minRadius=5, maxRadius=30
            )

            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                for (cx, cy, r) in circles:
                    # Verify it's bright enough to be a traffic light
                    roi_circle = hsv[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
                    if roi_circle.size == 0:
                        continue
                    avg_value = np.mean(roi_circle[:, :, 2])
                    if avg_value < 150:
                        continue

                    x1 = max(0, cx - r - 5)
                    y1 = max(0, cy - r - 5)
                    x2 = min(w, cx + r + 5)
                    y2 = min(int(h * 0.6), cy + r + 5)

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'state': state,
                        'conf': 0.7
                    })

        return detections

    def draw(self, frame, detections):
        """Draw traffic light detections on frame."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            state = det['state']
            conf = det['conf']
            color = self.STATE_COLORS.get(state, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{state} {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


# --- Training ---
def train_traffic_light_detector(data_yaml, epochs=50):
    """Fine-tune YOLOv8n for traffic light detection.

    data.yaml should have:
        nc: 3
        names: ['red', 'yellow', 'green']
    """
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, epochs=epochs, imgsz=640, batch=16,
                name="traffic_light", patience=10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Light Detection")
    parser.add_argument("--train", type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--demo", type=str, help="Video/image for demo")
    args = parser.parse_args()

    if args.train:
        train_traffic_light_detector(args.train, args.epochs)
    elif args.demo:
        det = TrafficLightDetector()
        cap = cv2.VideoCapture(args.demo)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            lights = det.detect(frame)
            frame = det.draw(frame, lights)
            cv2.imshow("Traffic Lights", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        parser.print_help()
