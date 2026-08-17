"""Deep analysis of WHY ego-lane pairing fails.

Questions to answer with real data, not assumptions:
1. After the current close kernel (5,41), how many components reach y > 80%?
2. What's the distribution of component heights?
3. If we lower the threshold to 70%, how many frames gain an ego pair?
4. What does the drivable-area mask look like relative to the lane lines?
   Can the DA edges substitute for a missing lane boundary?
5. On the highway clips (BDDA 100, 1003), are the dashes close enough that
   a taller close would bridge them without bridging adjacent lanes?

Run on 3 representative clips: Frankfurt (city), BDDA 100 (highway), BDDA 1003 (urban).
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR, PROJECT_ROOT
from trt_runner import TRTSeg

YOLOP_SZ = 384
device = torch.device("cuda")
yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)

OUT = PROJECT_ROOT / "output" / "deep_lane_analysis"
OUT.mkdir(parents=True, exist_ok=True)

videos = [
    ("frankfurt", "D:/carLane/downloads/frankfurt_720p_5min.mp4", 600),
    ("bdda100_highway", "D:/carLane/BDDA/test/camera_videos/100.mp4", 600),
    ("bdda1003_urban", "D:/carLane/BDDA/test/camera_videos/1003.mp4", 420),
]


def get_masks(path, n):
    """Return list of (frame, da_mask, ll_mask)."""
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
        da_t, ll_t = yolop.infer(t)
        da = cv2.resize(da_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((frame, da, ll))
    cap.release()
    return out, w, h


def analyze_components(ll, w, h, kx, ky, min_px=40):
    """Return list of (y_lo, y_hi, n_px, x_at_bottom) per component."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(ll, cv2.MORPH_CLOSE, kernel)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    comps = []
    for lab in range(1, n_lab):
        if stats[lab, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = np.nonzero(labels == lab)
        y_lo, y_hi = int(ys.min()), int(ys.max())
        # Where is this component at its lowest point (near bumper)?
        bot_mask = ys > (y_hi - 20)
        x_bot = float(xs[bot_mask].mean()) if bot_mask.any() else float(xs.mean())
        comps.append((y_lo, y_hi, len(xs), x_bot))
    return comps


def analyze_da_edges(da, w, h):
    """Find left and right edges of the drivable-area mask at each y-level.
    
    The DA mask is the full road surface. Its left and right boundaries at the
    bottom of the frame approximate the ego-lane boundaries when lane lines are
    absent or too fragmented.
    """
    left_edge = []
    right_edge = []
    for y in range(int(h * 0.5), h, 4):
        row = da[y]
        nz = np.nonzero(row)[0]
        if len(nz) > 10:
            left_edge.append((float(nz[0]), float(y)))
            right_edge.append((float(nz[-1]), float(y)))
    return left_edge, right_edge


def main():
    for label, path, n in videos:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        data, w, h = get_masks(path, n)
        
        # Q1: Component height distribution with current kernel (5,41)
        all_heights = []
        reach_80 = 0  # components reaching y > 0.8*h
        reach_70 = 0
        total_comps = 0
        frames_with_pair_80 = 0
        frames_with_pair_70 = 0
        
        # Q5: What about a much taller kernel?
        frames_with_pair_80_tall = 0
        
        # Q4: DA edge as substitute
        frames_with_da_pair = 0
        
        for i, (frame, da, ll) in enumerate(data):
            # Current kernel
            comps = analyze_components(ll, w, h, 5, 41, 40)
            total_comps += len(comps)
            heights = [c[1] - c[0] for c in comps]
            all_heights.extend(heights)
            
            # How many reach near bottom?
            centre = w / 2
            left_80 = [c for c in comps if c[1] > h * 0.80 and c[3] < centre]
            right_80 = [c for c in comps if c[1] > h * 0.80 and c[3] >= centre]
            if left_80 and right_80:
                frames_with_pair_80 += 1
            
            left_70 = [c for c in comps if c[1] > h * 0.70 and c[3] < centre]
            right_70 = [c for c in comps if c[1] > h * 0.70 and c[3] >= centre]
            if left_70 and right_70:
                frames_with_pair_70 += 1
            
            for c in comps:
                if c[1] > h * 0.80:
                    reach_80 += 1
                if c[1] > h * 0.70:
                    reach_70 += 1
            
            # Taller kernel (5, 101) — bridge dashes more aggressively
            comps_tall = analyze_components(ll, w, h, 5, 101, 40)
            left_80t = [c for c in comps_tall if c[1] > h * 0.80 and c[3] < centre]
            right_80t = [c for c in comps_tall if c[1] > h * 0.80 and c[3] >= centre]
            if left_80t and right_80t:
                frames_with_pair_80_tall += 1
            
            # DA edge approach
            le, re = analyze_da_edges(da, w, h)
            if len(le) > 5 and len(re) > 5:
                # Check the edges are reasonable (not wrapping around whole frame)
                l_x = np.mean([p[0] for p in le[-5:]])  # left edge at bottom
                r_x = np.mean([p[0] for p in re[-5:]])  # right edge at bottom
                width = r_x - l_x
                if w * 0.2 < width < w * 0.9:  # Plausible lane width
                    frames_with_da_pair += 1
            
            # Save diagnostic frames
            if i in (50, 200, 400):
                vis = frame.copy()
                vis[ll == 1] = (0, 255, 255)
                vis[da == 1] = np.clip(vis[da == 1].astype(np.int16) + [0, 30, 0], 0, 255).astype(np.uint8)
                # Draw DA edges
                for pts, col in ((le, (255, 0, 0)), (re, (0, 0, 255))):
                    if len(pts) > 2:
                        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, col, 3)
                cv2.putText(vis, f"{label} frame {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imwrite(str(OUT / f"{label}_f{i:04d}.jpg"), vis)
        
        n_f = len(data)
        print(f"\n  Components per frame: {total_comps/n_f:.1f}")
        if all_heights:
            ah = np.array(all_heights)
            print(f"  Component height (px): min={ah.min()} p25={np.percentile(ah,25):.0f} "
                  f"median={np.median(ah):.0f} p75={np.percentile(ah,75):.0f} max={ah.max()}")
            print(f"  Reaching y>80%h: {reach_80} ({reach_80/total_comps*100:.0f}% of components)")
            print(f"  Reaching y>70%h: {reach_70} ({reach_70/total_comps*100:.0f}% of components)")
        
        print(f"\n  Ego pair rate (current, threshold 80%): {frames_with_pair_80/n_f*100:.0f}%")
        print(f"  Ego pair rate (threshold lowered to 70%): {frames_with_pair_70/n_f*100:.0f}%")
        print(f"  Ego pair rate (tall kernel 5x101, threshold 80%): {frames_with_pair_80_tall/n_f*100:.0f}%")
        print(f"  DA-edge pair rate (drivable area boundaries): {frames_with_da_pair/n_f*100:.0f}%")
        
        print(f"\n  Best achievable ego coverage with each strategy:")
        best = max(frames_with_pair_80, frames_with_pair_70, 
                   frames_with_pair_80_tall, frames_with_da_pair)
        print(f"    Combined (lane lines OR DA edges): ~{best/n_f*100:.0f}%+")

    print(f"\n\nDiagnostic images saved to {OUT}")


if __name__ == "__main__":
    main()
