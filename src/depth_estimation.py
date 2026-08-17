"""Monocular Depth Estimation using Depth Anything V2.

Provides distance estimation from a single camera — no LiDAR/stereo needed.
Used for Forward Collision Warning (FCW) and distance-to-vehicle estimation.

Model: Depth Anything V2 (Small/Base/Large)
Paper: https://arxiv.org/abs/2406.09414
Repo: https://github.com/DepthAnything/Depth-Anything-V2

Zero training required — fully pretrained.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2."""

    def __init__(self, model_size="small", device="cuda"):
        """
        Args:
            model_size: 'small' (25M params), 'base' (97M), 'large' (335M)
            device: 'cuda' or 'cpu'
        """
        self.model = None
        self.device = device
        self.transform = None

        try:
            import torch
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor

            # Try loading via torch hub or transformers
            self._load_model(model_size)
        except ImportError as e:
            print(f"Depth estimation unavailable: {e}")
            print("Install: pip install torch torchvision transformers")

    def _load_model(self, model_size):
        """Load Depth Anything V2 model."""
        try:
            # Method 1: HuggingFace Transformers (easiest)
            from transformers import pipeline
            model_id = f"depth-anything/Depth-Anything-V2-{'Small' if model_size == 'small' else 'Base' if model_size == 'base' else 'Large'}-hf"
            self.pipe = pipeline("depth-estimation", model=model_id, device=0 if self.device == "cuda" else -1)
            self.use_pipeline = True
            print(f"Depth Anything V2 ({model_size}) loaded via HuggingFace")
        except Exception:
            # Method 2: Direct torch hub
            try:
                import torch
                self.model = torch.hub.load('DepthAnything/Depth-Anything-V2', f'depth_anything_v2_vit{model_size[0]}',
                                            pretrained=True)
                self.model.eval()
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
                self.use_pipeline = False
                print(f"Depth Anything V2 ({model_size}) loaded via torch hub")
            except Exception as e:
                print(f"Could not load depth model: {e}")
                self.pipe = None
                self.use_pipeline = True

    def estimate(self, frame):
        """Estimate depth map from a single BGR frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            depth_map: (H, W) float32 array, normalized 0-1 (0=far, 1=near)
            or None if model not loaded
        """
        if self.use_pipeline:
            if not hasattr(self, 'pipe') or self.pipe is None:
                return None
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.pipe(img)
            depth = np.array(result["depth"])
            # Normalize to 0-1
            depth = depth.astype(np.float32)
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            return depth
        else:
            if self.model is None:
                return None
            import torch
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Model expects specific input format
            with torch.no_grad():
                depth = self.model.infer_image(img)
            depth = depth.astype(np.float32)
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            return depth

    def get_distance_at_point(self, depth_map, x, y, focal_length=700, baseline_scale=10.0):
        """Estimate approximate distance at a pixel (relative, not metric).

        Args:
            depth_map: normalized depth map (0=far, 1=near)
            x, y: pixel coordinates
            focal_length: estimated camera focal length
            baseline_scale: scaling factor (tune for your setup)

        Returns:
            Relative distance score (lower = closer)
        """
        if depth_map is None:
            return float('inf')
        h, w = depth_map.shape
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        depth_value = depth_map[y, x]
        # Invert: higher depth value = closer, we want distance
        if depth_value > 0.01:
            return baseline_scale * (1.0 / depth_value)
        return float('inf')

    def get_distance_in_bbox(self, depth_map, bbox):
        """Get average distance within a bounding box.

        Args:
            depth_map: normalized depth map
            bbox: [x1, y1, x2, y2]

        Returns:
            Average relative distance (lower = closer)
        """
        if depth_map is None:
            return float('inf')
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return float('inf')

        avg_depth = np.mean(roi)
        if avg_depth > 0.01:
            return 10.0 * (1.0 / avg_depth)
        return float('inf')

    def colorize(self, depth_map):
        """Convert depth map to colored visualization.

        Returns:
            BGR colorized depth image
        """
        if depth_map is None:
            return None
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return colored
