"""YOLOP Inference Wrapper.

Handles lane detection, drivable area segmentation, and vehicle detection
using the pretrained YOLOP model (ONNX or PyTorch Hub).

YOLOP paper: https://arxiv.org/abs/2108.11250
YOLOP repo: https://github.com/hustvl/YOLOP
"""

import cv2
import numpy as np


class YOLOPInference:
    """Wrapper for YOLOP model inference.

    Supports two backends:
    - ONNX Runtime (faster, recommended for deployment)
    - PyTorch Hub (easier setup, heavier)
    """

    def __init__(self, model_path=None, backend="onnx", input_size=(640, 640),
                 conf_threshold=0.5, iou_threshold=0.45):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.backend = backend
        self.model = None

        if backend == "onnx" and model_path:
            self._load_onnx(model_path)
        elif backend == "pytorch":
            self._load_pytorch()
        else:
            print("YOLOP: No model loaded. Call load() with a valid path.")

    def _load_onnx(self, model_path):
        """Load YOLOP ONNX model."""
        import onnxruntime as ort
        self.model = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.model.get_inputs()[0].name
        print(f"YOLOP loaded (ONNX): {model_path}")

    def _load_pytorch(self):
        """Load YOLOP via PyTorch Hub."""
        import torch
        self.model = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("YOLOP loaded (PyTorch Hub)")

    def preprocess(self, frame):
        """Preprocess frame for YOLOP input.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, 640, 640), original shape
        """
        h, w = frame.shape[:2]
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # Add batch dim
        return img, (h, w)

    def infer(self, frame):
        """Run YOLOP inference on a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict with keys:
                'det_boxes': list of [x1, y1, x2, y2, conf, class] (vehicles)
                'da_seg': drivable area mask (H, W) binary
                'lane_seg': lane line mask (H, W) binary
        """
        img, orig_shape = self.preprocess(frame)
        h, w = orig_shape

        if self.backend == "onnx" and self.model:
            outputs = self.model.run(None, {self.input_name: img})
            # YOLOP ONNX outputs: [det_out, da_seg_out, ll_seg_out]
            det_out = outputs[0]      # Detection output
            da_seg_out = outputs[1]   # Drivable area segmentation
            ll_seg_out = outputs[2]   # Lane line segmentation

        elif self.backend == "pytorch" and self.model:
            import torch
            tensor = torch.from_numpy(img)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = self.model(tensor)
            det_out = det_out.cpu().numpy()
            da_seg_out = da_seg_out.cpu().numpy()
            ll_seg_out = ll_seg_out.cpu().numpy()
        else:
            # No model loaded — return empty results
            return {
                'det_boxes': [],
                'da_seg': np.zeros((h, w), dtype=np.uint8),
                'lane_seg': np.zeros((h, w), dtype=np.uint8)
            }

        # Post-process drivable area mask
        da_mask = np.argmax(da_seg_out[0], axis=0).astype(np.uint8)
        da_mask = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Post-process lane line mask
        ll_mask = np.argmax(ll_seg_out[0], axis=0).astype(np.uint8)
        ll_mask = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Post-process detections (NMS)
        boxes = self._postprocess_detections(det_out, orig_shape)

        return {
            'det_boxes': boxes,
            'da_seg': da_mask,
            'lane_seg': ll_mask
        }

    def _postprocess_detections(self, det_out, orig_shape):
        """Apply NMS and scale boxes back to original image size."""
        h, w = orig_shape
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]

        boxes = []
        if det_out is None or len(det_out) == 0:
            return boxes

        # det_out shape depends on export format
        # Typically: (batch, num_boxes, 6) where 6 = [x1, y1, x2, y2, conf, class]
        dets = det_out[0] if len(det_out.shape) == 3 else det_out

        for det in dets:
            if len(det) < 5:
                continue
            conf = det[4]
            if conf < self.conf_threshold:
                continue
            x1 = int(det[0] * scale_x)
            y1 = int(det[1] * scale_y)
            x2 = int(det[2] * scale_x)
            y2 = int(det[3] * scale_y)
            cls = int(det[5]) if len(det) > 5 else 0
            boxes.append([x1, y1, x2, y2, float(conf), cls])

        return boxes

    def draw_results(self, frame, results):
        """Draw YOLOP results on frame.

        Args:
            frame: BGR image
            results: dict from infer()

        Returns:
            Annotated frame
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        # Drivable area overlay (semi-transparent green)
        da_mask = results['da_seg']
        if da_mask.any():
            overlay = output.copy()
            overlay[da_mask == 1] = [0, 100, 0]  # Dark green
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

        # Lane lines (bright green)
        ll_mask = results['lane_seg']
        if ll_mask.any():
            output[ll_mask == 1] = [0, 255, 0]

        # Vehicle detection boxes (blue)
        for box in results['det_boxes']:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output, f"vehicle {conf:.0%}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

        return output


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import MODELS_DIR

    model_path = MODELS_DIR / "yolop.onnx"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Download it first: python src/download_data.py --models")
        print("Or use PyTorch Hub backend: --backend pytorch")
        sys.exit(1)

    yolop = YOLOPInference(str(model_path), backend="onnx")

    cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolop.infer(frame)
        output = yolop.draw_results(frame, results)
        cv2.imshow("YOLOP", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
