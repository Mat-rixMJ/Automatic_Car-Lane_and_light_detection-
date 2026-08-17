"""Traffic Sign Detector using YOLOv8.

Two-stage approach:
1. YOLOv8-nano detects sign regions (bounding boxes)
2. GTSRB CNN classifier identifies the specific sign type

This gives us both localization AND fine-grained classification (43 classes).
"""

import sys
from pathlib import Path

import cv2
import numpy as np


class SignDetector:
    """Detects and classifies traffic signs in frames.

    Uses YOLOv8n for detection + GTSRB CNN for classification.
    Falls back to color-based detection if YOLOv8 model is not available.
    """

    def __init__(self, detector_path=None, classifier_path=None, region="german",
                 conf_threshold=0.4, classifier_threshold=0.6):
        self.conf_threshold = conf_threshold
        self.classifier_threshold = classifier_threshold
        self.detector = None
        self.classifier = None

        # Load YOLOv8 detector if available
        if detector_path and Path(detector_path).exists():
            from ultralytics import YOLO
            self.detector = YOLO(detector_path)
            print(f"Sign detector loaded: {detector_path}")
        else:
            print("Sign detector: Using color-based fallback (no YOLOv8 model)")

        # Load CNN classifier
        if classifier_path and Path(classifier_path).exists():
            from sign_classifier import SignClassifier
            self.classifier = SignClassifier(classifier_path, region)
            print(f"Sign classifier loaded: {classifier_path}")

    def detect(self, frame):
        """Detect and classify traffic signs in a frame.

        Args:
            frame: BGR image

        Returns:
            list of dicts: [{'bbox': [x1,y1,x2,y2], 'class': str, 'conf': float}]
        """
        if self.detector:
            return self._detect_yolo(frame)
        else:
            return self._detect_color(frame)

    def _detect_yolo(self, frame):
        """Detect signs using YOLOv8, classify with CNN."""
        results = self.detector(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Crop and classify
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                class_name = "traffic sign"
                class_conf = conf

                if self.classifier:
                    cls_result = self.classifier.classify(roi)
                    if cls_result and cls_result[1] >= self.classifier_threshold:
                        class_name, class_conf = cls_result

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'conf': class_conf
                })

        return detections

    def _detect_color(self, frame):
        """Fallback: detect signs using color segmentation (red regions).

        Good enough for demos when you don't have a trained YOLOv8 sign detector.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color ranges (covers both ends of hue spectrum)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        # Blue signs (informational)
        mask3 = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

        red_mask = mask1 | mask2 | mask3

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800 or area > 50000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.6 < aspect_ratio < 1.5):
                continue

            roi = frame[y:y+h, x:x+w]
            class_name = "traffic sign"
            conf = 0.7  # Default confidence for color detection

            if self.classifier:
                cls_result = self.classifier.classify(roi)
                if cls_result:
                    class_name, conf = cls_result

            if conf >= self.classifier_threshold:
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'class': class_name,
                    'conf': conf
                })

        return detections

    def draw(self, frame, detections):
        """Draw sign detections on frame."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['conf']:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame


# --- Training script for sign detector ---
def train_sign_detector(data_yaml, epochs=50, model_size="yolov8n"):
    """Fine-tune YOLOv8n for traffic sign detection.

    Args:
        data_yaml: Path to YOLO-format dataset config
        epochs: Training epochs
        model_size: YOLOv8 variant (n/s/m/l/x)

    Expected data.yaml format:
        train: path/to/train/images
        val: path/to/val/images
        nc: 1  (or more if detecting sign sub-categories)
        names: ['traffic_sign']
    """
    from ultralytics import YOLO

    model = YOLO(f"{model_size}.pt")  # Load pretrained
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="sign_detector",
        patience=10
    )
    print(f"Training complete. Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import PROJECT_ROOT, MODELS_DIR, ensure_dirs

    parser = argparse.ArgumentParser(description="Traffic Sign Detection")
    parser.add_argument("--train", type=str, help="Path to data.yaml for training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--demo", type=str, help="Run demo on video/image")
    args = parser.parse_args()

    ensure_dirs()

    if args.train:
        train_sign_detector(args.train, args.epochs)
    elif args.demo:
        detector = SignDetector(
            detector_path=str(MODELS_DIR / "sign_detector.pt"),
            classifier_path=str(MODELS_DIR / "sign_classifier.keras")
        )
        cap = cv2.VideoCapture(args.demo)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            frame = detector.draw(frame, dets)
            cv2.imshow("Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        parser.print_help()
