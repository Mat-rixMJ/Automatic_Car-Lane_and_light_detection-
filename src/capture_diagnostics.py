"""Capture annotated screenshots AND measure what's wrong with them.

Every visual problem gets a number, so improvements are provable:

LANE / GREEN FILL
  spill      fraction of the green polygon NOT on YOLOP's road surface.
  width_cv   coefficient of variation of corridor width. High = breathing.
  jitter     mean frame-to-frame shift of the corridor centre, in px.

SIGNAL
  flip_rate  how often a light's state changes between consecutive frames.
  low_pos    lights detected below the horizon band (should be ~0).
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
from ultralytics import YOLO
from ego_corridor import EgoCorridor
from traffic_light import TrafficLightTracker

YOLOP_SZ = 384
OUT = PROJECT_ROOT / "output" / "diagnostics"

CLIPS = [
    ("frankfurt", r"D:\carLane\downloads\frankfurt_720p_5min.mp4", 900, 150),
    ("bdda100", r"D:\carLane\BDDA\test\camera_videos\100.mp4", 600, 120),
    ("bdda1003", r"D:\carLane\BDDA\test\camera_videos\1003.mp4", 420, 100),
]


def run(label, path, n_frames, shot_every):
    device = torch.device("cuda")
    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
    yolov8 = YOLO(str(MODELS_DIR / "yolov8n.engine"), task="detect")

    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(3)), int(cap.get(4))
    corridor = EgoCorridor(w, h)
    tracker = TrafficLightTracker(h)

    spills, widths, centres, jitters = [], [], [], []
    n = n_ok = 0
    raw_tl = kept_tl = low_pos = flips = 0
    prev_states = {}
    shots = []

    while n < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1

        img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)
        t = t.float().div_(255.0).contiguous()
        da_t, ll_t = yolop.infer(t)
        da = cv2.resize(da_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        ll_thick = cv2.dilate(ll, np.ones((3, 3), np.uint8), iterations=2)

        ok = corridor.update(da, lane_mask=ll_thick)
        if ok:
            n_ok += 1
        poly = corridor.polygon()

        if poly is not None:
            fill = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(fill, [poly], 1)
            area = int(fill.sum())
            if area > 0:
                on_road = int((fill & (da == 1).astype(np.uint8)).sum())
                spills.append(1.0 - on_road / area)
            wd = corridor.width_at_bottom()
            if wd:
                widths.append(wd)
            c = (corridor._left[-1] + corridor._right[-1]) / 2
            if centres:
                jitters.append(abs(c - centres[-1]))
            centres.append(c)

        cands = []
        for r in yolov8(frame, conf=0.2, verbose=False, imgsz=640):
            for b in r.boxes:
                if int(b.cls[0]) != 9:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                raw_tl += 1
                cands.append((x1, y1, x2, y2, float(b.conf[0])))
        lights = tracker.update(cands, frame)
        kept_tl += len(lights)
        cur = {}
        for (x1, y1, x2, y2, st, cf) in lights:
            if y1 > h * 0.55:
                low_pos += 1
            key = (x1 // 48, y1 // 48)
            cur[key] = st
            if key in prev_states and prev_states[key] != st:
                flips += 1
        prev_states = cur

        if n % shot_every == 0 and len(shots) < 8:
            vis = frame.copy()
            road = (da == 1)
            vis[road] = np.clip(vis[road].astype(np.int16) + (40, 0, 0), 0, 255).astype(np.uint8)
            vis[ll == 1] = (0, 220, 255)
            corridor.draw(vis, fill=True)
            for (x1, y1, x2, y2, st, cf) in lights:
                col = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255),
                       "GREEN": (0, 255, 0)}.get(st, (200, 200, 200))
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis, st, (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            info = [
                f"{label} f{n}",
                f"corridor={'YES' if ok else 'NO'}",
                f"spill={spills[-1]*100:.0f}%" if spills else "spill=-",
                f"width={widths[-1]:.0f}px ({widths[-1]/w*100:.0f}%)" if widths else "width=-",
                f"lights={len(lights)} (raw cands {len(cands)})",
            ]
            for i, s in enumerate(info):
                cv2.putText(vis, s, (10, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            p = OUT / f"{label}_f{n:05d}.jpg"
            cv2.imwrite(str(p), vis)
            shots.append(p.name)

    cap.release()

    print(f"\n{'='*56}\n  {label}  ({n} frames, {w}x{h})\n{'='*56}")
    print(f"  corridor available  {n_ok/n*100:5.0f}%")
    if spills:
        sp = np.array(spills)
        print(f"  green spill         {sp.mean()*100:5.0f}%  "
              f"(p75 {np.percentile(sp,75)*100:.0f}%, worst {sp.max()*100:.0f}%)")
    if widths:
        wd = np.array(widths)
        print(f"  corridor width      {wd.mean():5.0f}px = {wd.mean()/w*100:.0f}% of frame")
        print(f"  width variation     {wd.std()/max(wd.mean(),1)*100:5.0f}%")
    if jitters:
        jt = np.array(jitters)
        print(f"  centre jitter       {jt.mean():5.1f}px/frame (p95 {np.percentile(jt,95):.0f}px)")
    print(f"  --- signal ---")
    print(f"  raw candidates      {raw_tl:5d}")
    print(f"  kept after gates    {kept_tl:5d}  ({kept_tl/max(raw_tl,1)*100:.0f}% of raw)")
    print(f"  below horizon       {low_pos:5d}  ({low_pos/max(kept_tl,1)*100:.0f}% of kept)")
    print(f"  state flips         {flips:5d}  ({flips/max(kept_tl,1)*100:.0f}% of kept)")
    print(f"  screenshots: {', '.join(shots)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for args in CLIPS:
        run(*args)
    print(f"\n\nScreenshots in {OUT}")
