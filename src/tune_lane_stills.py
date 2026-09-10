"""Fast lane-tuning harness on still frames.

Runs the ego-seg model on every image in output/test_frames and, for each, saves
a 3-panel picture so we can SEE where lane detection needs work and tune the
post-processing without re-rendering whole videos:

  panel 1  RAW model mask (what the network actually predicts)
  panel 2  AFTER post-processing (component keep + lane clamp)
  panel 3  FINAL overlay on the frame (what the demo draws)

Run after any change to ego_seg_runner.py:
  python src/tune_lane_stills.py
Then open output/lane_tuning/*.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT

FRAMES = PROJECT_ROOT / "output" / "test_frames"
OUT = PROJECT_ROOT / "output" / "lane_tuning"


def raw_mask(model, frame, conf=0.35):
    """The model's raw ego mask, before any of our post-processing."""
    h, w = frame.shape[:2]
    r = model(frame, conf=conf, verbose=False, imgsz=640)[0]
    m = np.zeros((h, w), np.uint8)
    if r.masks is not None and len(r.masks) > 0:
        mm = r.masks.data.cpu().numpy()
        mm = (mm.max(axis=0) > 0.5).astype(np.uint8)
        m = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def tint(frame, mask, color):
    out = frame.copy()
    layer = np.zeros_like(frame)
    layer[mask == 1] = color
    cv2.addWeighted(layer, 0.5, out, 1.0, 0, dst=out)
    return out


def main():
    from ego_seg_runner import EgoSegRunner
    runner = EgoSegRunner(alpha=0.0)          # no temporal smoothing on stills
    model = runner.model

    OUT.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        print(f"No frames in {FRAMES}. Grab some first.")
        return

    for fp in frames:
        frame = cv2.imread(str(fp))
        h, w = frame.shape[:2]

        raw = raw_mask(model, frame)
        # run the real runner to get the post-processed mask + overlay
        runner._mask = None
        runner.update(frame)
        proc = runner.mask()
        if proc is None:
            proc = np.zeros((h, w), np.uint8)
        final = frame.copy()
        runner.draw(final)

        p1 = tint(frame, raw, (0, 165, 255))      # raw = orange
        cv2.putText(p1, "1. RAW model mask", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        p2 = tint(frame, proc, (0, 220, 0))        # processed = green
        cv2.putText(p2, "2. AFTER post-processing", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
        cv2.putText(final, "3. FINAL overlay", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        panel = np.vstack([p1, p2, final])
        panel = cv2.resize(panel, (w, int(h * 3 * (w / w))))  # keep width
        out_p = OUT / f"{fp.stem}_panels.jpg"
        cv2.imwrite(str(out_p), panel)
        cov_raw = raw.mean() * 100
        cov_proc = proc.mean() * 100
        print(f"  {fp.stem:22s} raw={cov_raw:4.1f}%  proc={cov_proc:4.1f}%  -> {out_p.name}")

    print(f"\nOpen {OUT} to review. Panels: raw / processed / final.")


if __name__ == "__main__":
    main()
