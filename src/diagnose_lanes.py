"""Verify LaneFitter on REAL YOLOP masks and dump side-by-side panels.

Reports, per sampled frame: raw lane pixel count, how many distinct lines were
fitted, whether an ego pair was found, the offset, and the drawn vertical span as
a fraction of frame height (the old version drew over the full height because it
extrapolated far beyond its data).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR, PROJECT_ROOT

VIDEO = r"D:\carLane\downloads\frankfurt_720p_5min.mp4"
OUT = PROJECT_ROOT / "output" / "lane_diag"
YOLOP_SZ = 384
SAMPLE_AT = [200, 900, 1800, 3000, 5000]
SCAN = 600


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    from trt_runner import TRTSeg
    from lane_fit import LaneFitter

    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
    cap = cv2.VideoCapture(VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fitter = LaneFitter(w, h)

    print(f"Frame {w}x{h}\n")
    print(f"{'frame':>6} {'lane px':>8} {'lines':>6} {'ego':>5} {'offset':>7} {'span%':>6}")

    n_fit = n_ego = n_frames = 0
    for idx in range(max(max(SAMPLE_AT), SCAN) + 1):
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)
        t = t.float().div_(255.0).contiguous()
        _, ll_t = yolop.infer(t)
        ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (w, h),
                        interpolation=cv2.INTER_NEAREST)

        ok = fitter.update(ll)
        st = fitter.stats()
        n_frames += 1
        n_fit += bool(ok)
        n_ego += bool(st["has_ego_pair"])

        if idx in SAMPLE_AT:
            npx = int((ll == 1).sum())
            span = 0.0
            if fitter.lines:
                lo = min(l.y_lo for l in fitter.lines)
                hi = max(l.y_hi for l in fitter.lines)
                span = (hi - lo) / h * 100
            off = st["offset"]
            print(f"{idx:6d} {npx:8d} {st['lines']:6d} "
                  f"{str(st['has_ego_pair']):>5} "
                  f"{(f'{off:.2f}' if off is not None else '-'):>7} {span:6.0f}")

            raw = frame.copy()
            raw[ll == 1] = (0, 255, 255)
            cv2.putText(raw, f"raw YOLOP mask ({npx} px)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            fit_vis = frame.copy()
            fitter.draw(fit_vis, fill=True)
            lbl = f"fitted: {st['lines']} lines, ego={st['has_ego_pair']}"
            cv2.putText(fit_vis, lbl, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if off is not None:
                cv2.putText(fit_vis, f"offset {off:+.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            panel = np.hstack([cv2.resize(v, (w // 2, h // 2)) for v in (raw, fit_vis)])
            cv2.imwrite(str(OUT / f"frame{idx:05d}.jpg"), panel)

    cap.release()
    print(f"\nOver {n_frames} frames: fitted {n_fit/n_frames*100:.0f}%, "
          f"ego pair {n_ego/n_frames*100:.0f}%")
    print(f"Panels in {OUT}")


if __name__ == "__main__":
    main()
