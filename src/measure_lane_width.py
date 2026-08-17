"""Measure the REAL ego-lane width from frames where actual lane lines paired up.

The DA fallback produces 47-66% wide corridors, which is clearly wrong. This
measures the width distribution from line-derived pairs only, to get an empirical
prior instead of another guess. Also measures width at several heights so the
prior can follow perspective.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR
from trt_runner import TRTSeg
from lane_fit import LaneFitter

YOLOP_SZ = 384
CLIPS = [
    ("frankfurt", r"D:\carLane\downloads\frankfurt_720p_5min.mp4", 1500),
    ("bdda100", r"D:\carLane\BDDA\test\camera_videos\100.mp4", 600),
    ("bdda1003", r"D:\carLane\BDDA\test\camera_videos\1003.mp4", 420),
    ("bdda1045", r"D:\carLane\BDDA\test\camera_videos\1045.mp4", 300),
]
# Heights to sample, as fraction of frame height
PROBES = [0.95, 0.90, 0.85, 0.80, 0.75]


def main():
    device = torch.device("cuda")
    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)

    per_probe = {p: [] for p in PROBES}
    centre_off = []

    for label, path, nf in CLIPS:
        cap = cv2.VideoCapture(path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fitter = LaneFitter(w, h)
        got = 0
        widths_bottom = []

        for _ in range(nf):
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)
            t = t.float().div_(255.0).contiguous()
            _, ll_t = yolop.infer(t)
            ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (w, h),
                            interpolation=cv2.INTER_NEAREST)

            # NOTE: no DA fallback — we only want genuine line-derived pairs
            fitter.update(ll)
            if fitter.left is None or fitter.right is None:
                continue

            # Both lines must actually exist at the probe height
            ok_all = True
            row = {}
            for p in PROBES:
                y = h * p
                if y > fitter.left.y_hi or y > fitter.right.y_hi:
                    ok_all = False
                    break
                if y < fitter.left.y_lo or y < fitter.right.y_lo:
                    ok_all = False
                    break
                lx = fitter.left.x_at(y)
                rx = fitter.right.x_at(y)
                wd = rx - lx
                if not (w * 0.08 < wd < w * 0.75):
                    ok_all = False
                    break
                row[p] = wd
            if not ok_all:
                continue

            got += 1
            for p, wd in row.items():
                per_probe[p].append(wd / w)          # normalised
            yb = h * 0.95
            lx, rx = fitter.left.x_at(yb), fitter.right.x_at(yb)
            widths_bottom.append((rx - lx) / w)
            centre_off.append(((lx + rx) / 2 - w / 2) / w)

        cap.release()
        if widths_bottom:
            wb = np.array(widths_bottom)
            print(f"{label:12s} {got:4d} line-pairs | width@95%h: "
                  f"median {np.median(wb)*100:.0f}% p25 {np.percentile(wb,25)*100:.0f}% "
                  f"p75 {np.percentile(wb,75)*100:.0f}%")
        else:
            print(f"{label:12s}    0 line-pairs")

    print("\nEmpirical ego-lane width by frame height (fraction of frame width):")
    print(f"  {'probe y':>8} {'n':>5} {'p10':>6} {'median':>7} {'p90':>6}")
    for p in PROBES:
        a = np.array(per_probe[p])
        if len(a) < 20:
            print(f"  {p*100:7.0f}% {len(a):5d}   (too few)")
            continue
        print(f"  {p*100:7.0f}% {len(a):5d} {np.percentile(a,10)*100:5.0f}% "
              f"{np.median(a)*100:6.0f}% {np.percentile(a,90)*100:5.0f}%")

    if centre_off:
        c = np.array(centre_off)
        print(f"\nLane centre offset from frame centre: "
              f"median {np.median(c)*100:+.0f}% of width, "
              f"p10 {np.percentile(c,10)*100:+.0f}%, p90 {np.percentile(c,90)*100:+.0f}%")

    # Fit width(y) so the prior can follow perspective
    ys, wds = [], []
    for p in PROBES:
        a = np.array(per_probe[p])
        if len(a) >= 20:
            ys.append(p)
            wds.append(float(np.median(a)))
    if len(ys) >= 3:
        coef = np.polyfit(ys, wds, 1)
        print(f"\nLinear prior: width_frac ~= {coef[0]:.3f} * (y/h) + {coef[1]:.3f}")
        for p in PROBES:
            print(f"   y={p*100:.0f}%h -> expected width {np.polyval(coef, p)*100:.0f}%")


if __name__ == "__main__":
    main()
