"""Validation pass — counts detections across a clip so quality is measurable.

Reports per-class detection rates, traffic-light state distribution, and lane/road
mask coverage. Also writes sample frames with the most detections for eyeballing.

Run:  python validate_pipeline.py --input <video> --seconds 60
"""

import sys
import time
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

YOLOP_SZ = 384
PROC_H = 512


def validate(video_path, seconds, save_samples=True):
    device = torch.device("cuda")

    from trt_runner import TRTSeg
    from ultralytics import YOLO
    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
    yolov8 = YOLO(str(MODELS_DIR / "yolov8n.engine"), task="detect")
    signdet = YOLO(str(MODELS_DIR / "german_sign_detector.engine"), task="detect")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(seconds * fps)

    veh_counter = Counter()
    tl_states = Counter()
    sign_classes = Counter()
    frames_with = Counter()
    road_cov, lane_cov = [], []
    best = []  # (n_detections, frame_idx) for sampling

    scale = PROC_H / h if h > PROC_H else 1.0
    sampled = 0

    print(f"Validating {w}x{h} @ {fps:.0f}fps — first {seconds}s ({n_frames} frames)")
    print("Running detection on EVERY frame (no skipping) for true rates...\n")

    t0 = time.time()
    idx = 0
    while idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        proc = cv2.resize(frame, (int(w * scale), PROC_H)) if scale != 1.0 else frame

        # --- Lanes / drivable area ---
        img = cv2.cvtColor(cv2.resize(proc, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
        da_t, ll_t = yolop.infer(t)
        da = da_t.squeeze().cpu().numpy()
        ll = ll_t.squeeze().cpu().numpy()
        road_cov.append(float((da == 1).mean()))
        lane_cov.append(float((ll == 1).mean()))
        if (ll == 1).sum() > 0:
            frames_with["lane"] += 1
        if (da == 1).sum() > 0:
            frames_with["road"] += 1

        # --- Vehicles + traffic lights ---
        n_det = 0
        got_veh = got_tl = False
        # Native frame at 640 — matches the pipeline, keeps small lights detectable
        for r in yolov8(frame, conf=0.2, verbose=False, imgsz=640):
            for box in r.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cid == 9:
                    bh, bw = y2 - y1, x2 - x1
                    aspect = bw / bh if bh else 1
                    if bh >= 12 and aspect < 1.0 and y1 < h * 0.75:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size:
                            th = max(1, roi.shape[0] // 3)
                            top, mid, bot = roi[:th], roi[th:2*th], roi[2*th:]
                            top_r = float(top[:, :, 2].mean()) - float(top[:, :, 1].mean())
                            bot_g = float(bot[:, :, 1].mean()) - float(bot[:, :, 2].mean())
                            mid_y = float(mid[:, :, 1].mean()) + float(mid[:, :, 2].mean())
                            if bot_g > 8:
                                st = "GREEN"
                            elif top_r > 8:
                                st = "RED"
                            elif mid_y > (top_r + bot_g) + 120:
                                st = "YELLOW"
                            else:
                                st = "UNKNOWN"
                            tl_states[st] += 1
                            got_tl = True
                            n_det += 1
                elif cid in (2, 3, 5, 7):
                    veh_counter[{2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}[cid]] += 1
                    got_veh = True
                    n_det += 1
        if got_veh:
            frames_with["vehicle"] += 1
        if got_tl:
            frames_with["traffic_light"] += 1

        # --- Signs ---
        got_sign = False
        for r in signdet(proc, conf=0.25, verbose=False, imgsz=480):
            for box in r.boxes:
                cid = int(box.cls[0])
                sign_classes[{0: "prohibitory", 1: "mandatory",
                              2: "danger", 3: "other"}.get(cid, "sign")] += 1
                got_sign = True
                n_det += 1
        if got_sign:
            frames_with["sign"] += 1

        best.append((n_det, idx, frame if save_samples and len(best) < 400 else None))

        if idx % 200 == 0:
            print(f"  {idx}/{n_frames} frames...")

    cap.release()
    total = idx
    dur = time.time() - t0

    print(f"\n{'='*58}")
    print(f"VALIDATION REPORT — {total} frames in {dur:.0f}s")
    print(f"{'='*58}")

    print("\nDetection presence (% of frames where class appears):")
    for k in ("road", "lane", "vehicle", "traffic_light", "sign"):
        pct = frames_with[k] / total * 100 if total else 0
        bar = "#" * int(pct / 2.5)
        print(f"  {k:14s} {pct:5.1f}%  {bar}")

    print(f"\nVehicles ({sum(veh_counter.values())} total detections):")
    for k, v in veh_counter.most_common():
        print(f"  {k:14s} {v:6d}  ({v/total:.2f}/frame)")

    print(f"\nTraffic lights ({sum(tl_states.values())} total detections):")
    for k, v in tl_states.most_common():
        print(f"  {k:14s} {v:6d}  ({v/max(1,sum(tl_states.values()))*100:5.1f}% of TL)")
    unknown = tl_states.get("UNKNOWN", 0)
    if sum(tl_states.values()):
        print(f"  -> state classified on {100 - unknown/sum(tl_states.values())*100:.1f}% of lights")

    print(f"\nSigns ({sum(sign_classes.values())} total detections):")
    for k, v in sign_classes.most_common():
        print(f"  {k:14s} {v:6d}  ({v/total:.2f}/frame)")

    print(f"\nSegmentation coverage (mean fraction of pixels):")
    print(f"  drivable area  {np.mean(road_cov)*100:5.1f}%  (std {np.std(road_cov)*100:.1f})")
    print(f"  lane lines     {np.mean(lane_cov)*100:5.1f}%  (std {np.std(lane_cov)*100:.1f})")

    if save_samples:
        out_dir = PROJECT_ROOT / "output" / "validation_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        top = sorted([b for b in best if b[2] is not None], key=lambda b: -b[0])[:5]
        for rank, (nd, fidx, fr) in enumerate(top, 1):
            cv2.imwrite(str(out_dir / f"top{rank}_frame{fidx}_{nd}det.jpg"), fr)
        print(f"\nSaved {len(top)} busiest frames to {out_dir}")

    print(f"\n{'='*58}")
    # Simple pass/fail gates so this is a real decision point
    gates = {
        "road detected >90% frames": frames_with["road"] / total > 0.90,
        "lanes detected >80% frames": frames_with["lane"] / total > 0.80,
        "vehicles detected >50% frames": frames_with["vehicle"] / total > 0.50,
        "TL state classified >70%": (sum(tl_states.values()) == 0
                                     or (1 - unknown / sum(tl_states.values())) > 0.70),
    }
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{'='*58}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=r"D:\carLane\downloads\frankfurt_720p_5min.mp4")
    p.add_argument("--seconds", type=int, default=60)
    a = p.parse_args()
    validate(a.input, a.seconds)
