"""Before/after lane eval: heuristic corridor vs learned ego-seg, real IoU.

The honest comparison for judges. Both methods are scored against the SAME
held-out BDD100K ground-truth ego-lane masks (class 0 = direct drivable), which
neither the corridor (hand-tuned) nor — critically — the seg model (trained only
on the 8k train split) has seen. Metric: mask IoU vs ground truth.

  corridor: YOLOP drivable mask -> EgoCorridor polygon -> filled mask
  ego-seg : the trained model's ego mask

Reports mean IoU, and the fraction of frames each "finds" the lane at IoU>=0.5.

  python src/eval_lane_iou.py --n 300
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

VAL_IMG = PROJECT_ROOT / "BDD100K" / "images" / "val"
VAL_MASK = PROJECT_ROOT / "BDD100K" / "labels" / "drivable_labels" / "masks" / "val"
YOLOP_SZ = 384


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else (1.0 if a.sum() == b.sum() == 0 else 0.0)


def gt_ego(mask_path):
    """Ground-truth ego mask = class 0 (direct drivable) in the BDD mask."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)


def corridor_mask(da_full, frame_w, frame_h, EgoCorridor):
    c = EgoCorridor(frame_w, frame_h)
    c.update(da_full)
    poly = c.polygon()
    m = np.zeros((frame_h, frame_w), np.uint8)
    if poly is not None:
        cv2.fillPoly(m, [poly], 1)
    return m


def run(n):
    import torch
    from ego_corridor import EgoCorridor
    from ego_seg_runner import EgoSegRunner
    from trt_runner import TRTSeg

    device = torch.device("cuda")
    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
    seg = EgoSegRunner(alpha=0.0)         # no temporal smoothing on stills

    masks = sorted(VAL_MASK.glob("*.png"))[:n]
    iou_corr, iou_seg = [], []
    hit_corr = hit_seg = 0

    for i, mp in enumerate(masks, 1):
        img_p = VAL_IMG / f"{mp.stem}.jpg"
        if not img_p.exists():
            continue
        frame = cv2.imread(str(img_p))
        h, w = frame.shape[:2]
        gt = gt_ego(mp)
        if gt is None:
            continue

        # YOLOP drivable mask for the corridor baseline
        img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
        da_t, _ = yolop.infer(t)
        da = cv2.resize(da_t.squeeze().cpu().numpy().astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)
        cm = corridor_mask(da, w, h, EgoCorridor)

        # learned ego-seg (reset EMA each still)
        seg._mask = None
        seg.update(frame)
        sm = seg.mask()
        if sm is None:
            sm = np.zeros((h, w), np.uint8)

        ic, is_ = iou(cm, gt), iou(sm, gt)
        iou_corr.append(ic); iou_seg.append(is_)
        hit_corr += ic >= 0.5; hit_seg += is_ >= 0.5
        if i % 50 == 0:
            print(f"  {i}/{len(masks)}")

    n_ok = len(iou_seg)
    print(f"\n{'='*58}\n  LANE EVAL  ({n_ok} held-out BDD val frames)\n{'='*58}")
    print(f"  {'method':22s} {'mean IoU':>10s} {'IoU>=0.5':>10s}")
    print("  " + "-" * 44)
    print(f"  {'EgoCorridor (heuristic)':22s} {np.mean(iou_corr):10.3f} "
          f"{hit_corr/n_ok*100:9.0f}%")
    print(f"  {'ego-seg (trained)':22s} {np.mean(iou_seg):10.3f} "
          f"{hit_seg/n_ok*100:9.0f}%")
    print("  " + "-" * 44)
    gain = (np.mean(iou_seg) - np.mean(iou_corr))
    print(f"  IoU gain from training: {gain:+.3f} "
          f"({gain/max(np.mean(iou_corr),1e-6)*100:+.0f}%)")
    print(f"{'='*58}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()
    run(a.n)
