"""Ultra-Fast-Lane-Detection (UFLD v1, ResNet18, TuSimple) — direct ego-lane lines.

This is the README's named lane upgrade: a model that outputs lane BOUNDARIES
directly, instead of carving a corridor out of YOLOP's drivable-area mask. The
corridor approach was model-based (it assumed the ego lane is the road straight
ahead) and could not represent a lane change; a learned lane detector can.

Why UFLD v1 and not v2: v1's decode is a plain per-row softmax + expectation over
grid cells — short, verifiable, no custom CUDA ops for inference. v2's hybrid-anchor
decode is intricate and easy to get subtly wrong. v1 is the lazy-correct choice.

Weights: cfzd/Ultra-Fast-Lane-Detection, CULane ResNet18, MIT-licensed.
Config (read off the checkpoint, not assumed): cls head = 201*18*4 = 14472, i.e.
  GRIDING = 200 (+1 "no lane"), ROW_ANCHORS = 18, LANES = 4. Input 288x800 RGB.

CULane, not TuSimple: TuSimple row anchors (64-284) are defined for a highway
camera and, on this dashcam footage, land lane points above the horizon (verified
visually — they fell in the sky). CULane anchors cover the near/lower field where
dashcam lanes actually are, so it generalises to this footage. This is why the
HF LiteRT reference and most dashcam UFLD deployments use CULane.

The network is a stock torchvision ResNet18 backbone (keys match after a 'model.'
prefix) + a 1x1 Conv 'pool' + a 2-layer Linear 'cls' head, so it rebuilds here
without cloning the training repo (which needs Nvidia DALI + a custom interp op,
both training-only).

ponytail: CULane weights generalise to dashcam far better than TuSimple, but are
still trained on Chinese urban/highway footage. Ceiling — non-standard markings
and heavy occlusion are weaker. Upgrade path = fine-tune on BDD100K lane labels.

Self-check (no GPU needed):  python src/ufld_lane.py --selftest
Export ONNX:                 python src/ufld_lane.py --export
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR

# --- config, taken from the checkpoint head shape 14472 = 201*18*4 -----------
GRIDING = 200                 # grid cells across; +1 for the "no lane" bin
ROW_ANCHORS_N = 18
LANES = 4
CLS_DIM = (GRIDING + 1, ROW_ANCHORS_N, LANES)   # (201, 18, 4)
IN_H, IN_W = 288, 800

PTH_PATH = MODELS_DIR / "culane_18.pth"
ONNX_PATH = MODELS_DIR / "ufld_culane_18.onnx"

# CULane evaluation row anchors, defined in a 590px-high image. The decode maps
# grid rows to these y-values, so they must match the protocol the weights were
# trained on. These 18 anchors span the near/lower field (121..287 of 590),
# i.e. the road surface ahead of the car — the canonical UFLD CULane anchors.
CULANE_ROW_ANCHORS = [
    121, 131, 141, 150, 160, 170, 180, 189, 199, 209,
    219, 228, 238, 248, 258, 267, 277, 287,
]
CULANE_IMG_H = 590            # row anchors are defined in a 590-high image


def build_model():
    """Rebuild UFLD v1 (ResNet18) as an inference module and load MIT weights."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    class UFLD(nn.Module):
        def __init__(self):
            super().__init__()
            bb = resnet18(weights=None)
            # Expose the backbone under the same names the checkpoint uses
            # (model.conv1, model.bn1, model.layer1..4), dropping avgpool/fc.
            self.model = nn.Module()
            self.model.conv1 = bb.conv1
            self.model.bn1 = bb.bn1
            self.model.relu = bb.relu
            self.model.maxpool = bb.maxpool
            self.model.layer1 = bb.layer1
            self.model.layer2 = bb.layer2
            self.model.layer3 = bb.layer3
            self.model.layer4 = bb.layer4
            self.pool = nn.Conv2d(512, 8, 1)
            self.cls = nn.Sequential(
                nn.Linear(1800, 2048), nn.ReLU(),
                nn.Linear(2048, int(np.prod(CLS_DIM))),
            )

        def forward(self, x):
            m = self.model
            x = m.relu(m.bn1(m.conv1(x)))
            x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            x = self.pool(x).view(-1, 1800)
            return self.cls(x).view(-1, *CLS_DIM)

    model = UFLD().eval()
    sd = torch.load(PTH_PATH, map_location="cpu")
    sd = sd.get("model", sd)
    # Strip aux-head keys (training only) and load the rest strictly.
    sd = {k: v for k, v in sd.items() if not k.startswith("aux")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    hard_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if hard_missing:
        raise RuntimeError(f"weight load missing real keys: {hard_missing[:6]}")
    return model


def export_onnx():
    import torch
    if ONNX_PATH.exists():
        print(f"  ONNX exists: {ONNX_PATH.name}")
        return
    model = build_model()
    dummy = torch.randn(1, 3, IN_H, IN_W)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["images"], output_names=["cls"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  ONNX: {ONNX_PATH.name} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")


# The model is fed a CROP of the frame (the road band), not the whole frame.
# UFLD CULane was trained on road-dominant dashcam images; feeding a full frame
# with a big sky/building region makes the row-anchor head predict lanes up in
# the buildings (verified visually — the points arched across the skyline). The
# crop restores the road-dominant framing the model expects. The band is a
# fraction of frame height; the anchors then map into that band, not the frame.
CROP_TOP_FRAC = 0.40          # start the road band this far down the frame
CROP_BOT_FRAC = 1.00          # ...to the bottom (bonnet is a small cost)


# --- decode: raw net output -> lane points in the ORIGINAL frame -------------
def decode(out, frame_w, frame_h, crop_top_frac=CROP_TOP_FRAC,
           crop_bot_frac=CROP_BOT_FRAC):
    """out: (201, 18, 4) logits. Returns list of 4 lanes, each a list of (x,y)
    points in the ORIGINAL frame, empty where the model says 'no lane'.

    Per lane, per row: softmax over the 200 grid cells, take the expectation to
    get a sub-pixel column; if the argmax over all 201 bins is the last bin
    (index 200 = 'no lane'), that row has no lane point. Standard UFLD decode.

    The network saw only the crop band [crop_top_frac, crop_bot_frac] of the
    frame height, so row anchors (defined over the crop) map back into that band.
    """
    out = np.asarray(out, dtype=np.float64)
    # rows are stored top-first in the net; anchors are listed bottom-up in
    # eval, so reverse rows to align (matches the reference decode).
    out = out[:, ::-1, :]
    prob = np.exp(out[:GRIDING]) / np.exp(out[:GRIDING]).sum(0, keepdims=True)
    idx = (np.arange(GRIDING) + 1).reshape(-1, 1, 1)
    loc = (prob * idx).sum(0)                       # (18, 4), 1-based grid units
    argmax = out.argmax(0)                          # (18, 4)
    has_lane = argmax != GRIDING                    # False where 'no lane' wins

    # x spans the full frame width (the crop keeps full width). loc is 1-based
    # (idx started at 1), so subtract 1 to a 0-based cell before scaling.
    col_to_x = frame_w / (GRIDING - 1.0)
    band_top = crop_top_frac * frame_h
    band_h = (crop_bot_frac - crop_top_frac) * frame_h
    lanes = []
    for lane_i in range(LANES):
        pts = []
        for row_i, y_anchor in enumerate(reversed(CULANE_ROW_ANCHORS)):
            if not has_lane[row_i, lane_i]:
                continue
            col = loc[row_i, lane_i] - 1.0          # back to 0-based grid cell
            if col < 0:
                continue
            x = col * col_to_x
            # anchor is a y within the crop (0..CULANE_IMG_H) -> map into band
            y = band_top + (y_anchor / CULANE_IMG_H) * band_h
            pts.append((float(x), float(y)))
        lanes.append(pts)
    return lanes


class UFLDLaneDetector:
    """Runtime wrapper: preprocess -> onnxruntime -> decode. GPU if available."""

    def __init__(self, onnx_path=ONNX_PATH):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self._mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

    def _pre(self, frame_bgr):
        import cv2
        h = frame_bgr.shape[0]
        y0 = int(h * CROP_TOP_FRAC)
        y1 = int(h * CROP_BOT_FRAC)
        crop = frame_bgr[y0:y1]                     # road band, full width
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IN_W, IN_H)).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self._mean) / self._std
        return img[None].astype(np.float32)

    def detect(self, frame_bgr):
        """Return 4 lanes as point lists in the input frame's own coordinates."""
        h, w = frame_bgr.shape[:2]
        out = self.sess.run(None, {self.inp: self._pre(frame_bgr)})[0][0]
        return decode(out, w, h)


# --------------------------------------------------------------------------
def _selftest():
    """Weight load + shape + decode maths, all on CPU."""
    import torch
    assert int(np.prod(CLS_DIM)) == 14472, "config doesn't match checkpoint head"
    assert len(CULANE_ROW_ANCHORS) == ROW_ANCHORS_N, "row anchor count mismatch"

    if PTH_PATH.exists():
        model = build_model()
        with torch.no_grad():
            y = model(torch.randn(1, 3, IN_H, IN_W))
        assert tuple(y.shape) == (1, *CLS_DIM), y.shape
        print(f"  weights loaded, forward output {tuple(y.shape)} OK")
    else:
        print(f"  (skipping weight load — {PTH_PATH.name} not present)")

    # Decode maths: craft logits where lane 0 points hard at grid col 50 for
    # every row, lane 1 says 'no lane' everywhere. Expect lane0 x≈middle, lane1 empty.
    fake = np.full(CLS_DIM, -10.0, np.float64)
    fake[50, :, 0] = 10.0            # lane 0 -> column 50 at every row
    fake[GRIDING, :, 1] = 10.0       # lane 1 -> 'no lane' bin
    lanes = decode(fake, frame_w=1280, frame_h=720)
    assert len(lanes) == 4
    assert len(lanes[0]) == ROW_ANCHORS_N, f"lane0 rows {len(lanes[0])}"
    xs = [p[0] for p in lanes[0]]
    # hard peak at grid index 50 -> 0-based cell 50 -> x = 50/(GRIDING-1)*w
    exp_x = 50 * (1280 / (GRIDING - 1.0))
    assert all(abs(x - exp_x) < 1.0 for x in xs), f"decode x off: {xs[:3]} vs {exp_x}"
    assert len(lanes[1]) == 0, f"lane1 should be empty, got {len(lanes[1])}"
    # y must land inside the road band [CROP_TOP_FRAC, CROP_BOT_FRAC] of the
    # frame, never up in the sky — this is the assertion the sky-placement bug
    # would have failed.
    ys = [p[1] for p in lanes[0]]
    band_lo = CROP_TOP_FRAC * 720
    band_hi = CROP_BOT_FRAC * 720
    assert min(ys) < max(ys), "y should vary across rows"
    assert min(ys) >= band_lo - 1 and max(ys) <= band_hi + 1, \
        f"y {min(ys):.0f}-{max(ys):.0f} outside road band {band_lo:.0f}-{band_hi:.0f}"
    print("  decode maths OK: column->x, no-lane bin, row->band-y all correct")
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="UFLD lane detector")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--export", action="store_true", help="export ONNX from .pth")
    a = p.parse_args()
    if a.export:
        export_onnx()
    else:
        _selftest()
