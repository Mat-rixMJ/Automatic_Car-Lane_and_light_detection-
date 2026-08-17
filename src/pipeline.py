"""Unified Perception Pipeline.

Orchestrates all modules:
- YOLOP: lane detection + drivable area + vehicle detection
- Sign Detection: traffic sign localization + GTSRB classification
- Traffic Light Detection: light state (red/yellow/green)

Outputs a single annotated frame with all detections overlaid.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class PerceptionPipeline:
    """Full autonomous driving perception pipeline (hybrid architecture)."""

    def __init__(self, config=None):
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import load_config, MODELS_DIR

        if config is None:
            config = load_config()

        self.config = config
        models_cfg = config.get("models", {})

        # --- Load YOLOP ---
        yolop_path = Path(models_cfg.get("yolop", "models/yolop.onnx"))
        if not yolop_path.is_absolute():
            yolop_path = Path(__file__).parent.parent / yolop_path

        from yolop_inference import YOLOPInference
        if yolop_path.exists():
            yolop_cfg = config.get("yolop", {})
            self.yolop = YOLOPInference(
                str(yolop_path), backend="onnx",
                conf_threshold=yolop_cfg.get("conf_threshold", 0.5),
                iou_threshold=yolop_cfg.get("iou_threshold", 0.45)
            )
        else:
            # Try PyTorch Hub fallback
            print("YOLOP ONNX not found, trying PyTorch Hub...")
            self.yolop = YOLOPInference(backend="pytorch")

        # --- Load Sign Detector ---
        sign_cfg = config.get("sign_detection", {})
        det_path = models_cfg.get("sign_detector", "models/sign_detector.pt")
        cls_path = models_cfg.get("sign_classifier", "models/sign_classifier.keras")
        if not Path(det_path).is_absolute():
            det_path = str(Path(__file__).parent.parent / det_path)
        if not Path(cls_path).is_absolute():
            cls_path = str(Path(__file__).parent.parent / cls_path)

        from sign_detection import SignDetector
        self.sign_detector = SignDetector(
            detector_path=det_path,
            classifier_path=cls_path,
            region=config.get("region", "german"),
            conf_threshold=sign_cfg.get("conf_threshold", 0.4),
            classifier_threshold=sign_cfg.get("classifier_threshold", 0.6)
        )

        # --- Load Traffic Light Detector ---
        tl_cfg = config.get("traffic_light", {})
        tl_path = models_cfg.get("traffic_light", "models/traffic_light.pt")
        if not Path(tl_path).is_absolute():
            tl_path = str(Path(__file__).parent.parent / tl_path)

        # Fallback to YOLOv8n COCO (class 9 = traffic light) if no dedicated model
        coco_path = models_cfg.get("yolov8n_coco", "models/yolov8n.pt")
        if not Path(coco_path).is_absolute():
            coco_path = str(Path(__file__).parent.parent / coco_path)

        from traffic_light_detection import TrafficLightDetector
        self.traffic_light_detector = TrafficLightDetector(
            model_path=tl_path,
            use_coco_model=coco_path,
            conf_threshold=tl_cfg.get("conf_threshold", 0.5)
        )

        # --- Depth Estimation (for collision warning) ---
        self.depth_estimator = None
        if config.get("adas", {}).get("enable_depth", True):
            try:
                from depth_estimation import DepthEstimator
                depth_size = config.get("adas", {}).get("depth_model", "small")
                self.depth_estimator = DepthEstimator(model_size=depth_size)
            except Exception as e:
                print(f"Depth estimation disabled: {e}")

        # --- ADAS Module ---
        from adas_features import AdasModule
        self.adas = AdasModule()

        # State for HUD
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame):
        """Run all perception modules on a frame.

        Args:
            frame: BGR image

        Returns:
            Annotated frame with all detections
        """
        output = frame.copy()

        # 1. YOLOP — lanes, drivable area, vehicles
        yolop_results = self.yolop.infer(frame)
        output = self.yolop.draw_results(output, yolop_results)

        # 2. Traffic signs
        sign_detections = self.sign_detector.detect(frame)
        output = self.sign_detector.draw(output, sign_detections)

        # 3. Traffic lights
        light_detections = self.traffic_light_detector.detect(frame)
        output = self.traffic_light_detector.draw(output, light_detections)

        # 4. Depth estimation (optional, for collision warning)
        depth_map = None
        if self.depth_estimator:
            depth_map = self.depth_estimator.estimate(frame)

        # 5. ADAS — collision warning, lane departure, speed limit
        adas_state = self.adas.update(
            yolop_results, sign_detections, light_detections,
            depth_map=depth_map, frame_shape=frame.shape
        )

        # 6. Draw ADAS dashboard
        output = self.adas.draw_dashboard(output)

        # 7. Top HUD bar
        output = self._draw_hud(output, yolop_results, sign_detections, light_detections)

        return output

    def _draw_hud(self, frame, yolop_results, signs, lights):
        """Draw heads-up display with summary info."""
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "CarLaneI - Hybrid Perception",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Bottom status bar
        status_y = h - 15
        status_items = []
        status_items.append(f"Vehicles: {len(yolop_results['det_boxes'])}")
        status_items.append(f"Signs: {len(signs)}")
        if lights:
            light_state = lights[0]['state'].upper()
            status_items.append(f"Light: {light_state}")

        status_text = " | ".join(status_items)
        cv2.putText(frame, status_text, (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run_video(self, source, output_path=None):
        """Run pipeline on video file or webcam."""
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            print(f"Error: Cannot open '{source}'")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing: {source} ({width}x{height} @ {fps}fps)")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            if writer:
                writer.write(result)
            cv2.imshow("CarLaneI Pipeline", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        print(f"Done: {self.frame_count} frames, {elapsed:.1f}s, {self.frame_count/elapsed:.1f} FPS")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import ensure_dirs

    parser = argparse.ArgumentParser(description="CarLaneI Perception Pipeline")
    parser.add_argument("--input", type=str, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--webcam", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    pipeline = PerceptionPipeline()

    if args.webcam:
        pipeline.run_video(0, args.output)
    elif args.input:
        pipeline.run_video(args.input, args.output)
    else:
        parser.print_help()
