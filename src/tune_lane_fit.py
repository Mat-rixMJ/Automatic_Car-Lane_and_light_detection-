"""Measure why lane components get rejected, and sweep the dash-joining kernel.

The ego-pair rate was 42%. Before tuning anything, count what the filters throw
away and how much a taller morphological close (which joins dashed markings into
one component) actually recovers.
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR

VIDEO = r"D:\carLane\downloads\frankfurt_720p_5min.mp4"
YOLOP_SZ = 384
N = 400
W, H = 1280, 720


def masks(n):
    device = torch.device("cuda")
    from trt_runner import TRTSeg
    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
    cap = cv2.VideoCapture(VIDEO)
    out = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)
        t = t.float().div_(255.0).contiguous()
        _, ll_t = yolop.infer(t)
        out.append(cv2.resize(ll_t.squeeze().cpu().numpy(), (W, H),
                              interpolation=cv2.INTER_NEAREST))
    cap.release()
    return out


def why_rejected(ms, kx, ky, min_px, min_span, max_resid):
    r = Counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    for m in ms:
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        for lab in range(1, n_lab):
            r["total"] += 1
            if stats[lab, cv2.CC_STAT_AREA] < min_px:
                r["too_few_px"] += 1
                continue
            if stats[lab, cv2.CC_STAT_HEIGHT] < min_span:
                r["too_short"] += 1
                continue
            ys, xs = np.nonzero(labels == lab)
            y_lo, y_hi = int(ys.min()), int(ys.max())
            order = 2 if (y_hi - y_lo) > 60 else 1
            try:
                coef = np.polyfit(ys, xs, order)
            except Exception:
                r["fit_error"] += 1
                continue
            if order == 1:
                coef = np.array([0.0, coef[0], coef[1]])
            if np.abs(np.polyval(coef, ys) - xs).mean() > max_resid:
                r["bad_residual"] += 1
                continue
            r["accepted"] += 1
    return r


def sweep(ms):
    from lane_fit import LaneFitter
    print(f"\n{'kernel':>10} {'minpx':>6} {'span':>5} {'resid':>6} "
          f"{'fitted%':>8} {'ego%':>6} {'lines/f':>8} {'span%':>6}")
    configs = [
        ((3, 9), 45, 18, 12),      # current
        ((3, 25), 45, 18, 12),
        ((5, 41), 45, 18, 12),
        ((5, 41), 45, 18, 20),
        ((5, 61), 40, 15, 20),
        ((7, 81), 40, 15, 25),
    ]
    for (kx, ky), mp, msp, mr in configs:
        f = LaneFitter(W, H)
        f.MIN_PX, f.MIN_SPAN = mp, msp
        n_fit = n_ego = 0
        tot_lines = 0
        spans = []
        for m in ms:
            closed_kernel = (kx, ky)
            ok = f.update(m, kernel=closed_kernel, max_resid=mr)
            n_fit += bool(ok)
            st = f.stats()
            n_ego += bool(st["has_ego_pair"])
            tot_lines += st["lines"]
            if f.lines:
                spans.append((max(l.y_hi for l in f.lines)
                              - min(l.y_lo for l in f.lines)) / H * 100)
        n = len(ms)
        print(f"{str((kx,ky)):>10} {mp:6d} {msp:5d} {mr:6d} "
              f"{n_fit/n*100:7.0f}% {n_ego/n*100:5.0f}% "
              f"{tot_lines/n:8.1f} {np.mean(spans) if spans else 0:6.0f}")


if __name__ == "__main__":
    print(f"Loading {N} real YOLOP masks...")
    ms = masks(N)
    print(f"Got {len(ms)}")

    print("\nComponent rejection breakdown at current settings:")
    r = why_rejected(ms, 3, 9, 45, 18, 12)
    tot = r["total"] or 1
    for k in ("accepted", "too_few_px", "too_short", "bad_residual", "fit_error"):
        print(f"  {k:14s} {r[k]:6d}  ({r[k]/tot*100:4.1f}%)")

    sweep(ms)
