"""Diagnose why traffic lights are missed and signs are mislabelled.

Part A: traffic-light box size distribution + detection counts at different
        input resolutions / crops, to see if we're losing small objects to
        downscaling rather than to model weakness.
Part B: sign classifier confidence + top1-vs-top2 margin on real detected crops,
        to see whether out-of-distribution signs (e.g. U-turn, absent from GTSRB)
        can be rejected by a margin gate.
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
FRAMES = 400


def part_a():
    from ultralytics import YOLO
    det = YOLO(str(MODELS_DIR / "yolov8n.engine"), task="detect")
    det_pt = YOLO(str(MODELS_DIR / "yolov8n.pt"))  # flexible imgsz for comparison

    cap = cv2.VideoCapture(VIDEO)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    sizes = []
    counts = Counter()
    frames_with = Counter()

    print(f"PART A — traffic light detection vs input strategy ({FRAMES} frames, {w}x{h})")
    print("Strategies:")
    print("  proc512  : downscale to 512p, detect (what the pipeline does now)")
    print("  full720  : detect on native 1280x720")
    print("  crop_up  : crop upper-centre 640x360 at NATIVE res, detect")
    print()

    for i in range(FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        # Strategy 1: current pipeline — downscale to 512p
        proc = cv2.resize(frame, (int(w * 512 / h), 512))
        n1 = 0
        for r in det_pt(proc, conf=0.2, verbose=False, imgsz=480):
            for b in r.boxes:
                if int(b.cls[0]) == 9:
                    n1 += 1
        counts["proc512"] += n1
        if n1:
            frames_with["proc512"] += 1

        # Strategy 2: native 720p
        n2 = 0
        for r in det_pt(frame, conf=0.2, verbose=False, imgsz=640):
            for b in r.boxes:
                if int(b.cls[0]) == 9:
                    n2 += 1
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    sizes.append(y2 - y1)
        counts["full720"] += n2
        if n2:
            frames_with["full720"] += 1

        # Strategy 3: upper-centre crop at native resolution
        # Lights hang above the road, usually centre-ish
        cy0, cy1 = 0, int(h * 0.6)
        cx0, cx1 = int(w * 0.15), int(w * 0.85)
        crop = frame[cy0:cy1, cx0:cx1]
        n3 = 0
        for r in det_pt(crop, conf=0.2, verbose=False, imgsz=640):
            for b in r.boxes:
                if int(b.cls[0]) == 9:
                    n3 += 1
        counts["crop_up"] += n3
        if n3:
            frames_with["crop_up"] += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{FRAMES}...")

    cap.release()
    n = min(FRAMES, i + 1)

    print(f"\n  {'strategy':10s} {'detections':>11s} {'frames w/ TL':>13s}")
    for k in ("proc512", "full720", "crop_up"):
        print(f"  {k:10s} {counts[k]:11d} {frames_with[k]/n*100:12.1f}%")

    if sizes:
        a = np.array(sizes)
        print(f"\n  TL box height in native px (n={len(a)}):")
        print(f"    min {a.min():.0f}  p25 {np.percentile(a,25):.0f}  "
              f"median {np.median(a):.0f}  p75 {np.percentile(a,75):.0f}  max {a.max():.0f}")
        print(f"    fraction under 25px: {(a<25).mean()*100:.0f}%")
        # What those become after the pipeline's downscale+letterbox
        eff = a * (512 / 720) * (480 / (1280 * 512 / 720))
        print(f"    after proc512 + 480 letterbox, median becomes {np.median(eff):.1f}px")


def part_b():
    from sign_classifier import SignClassifier
    from ultralytics import YOLO
    import json

    names = json.loads((Path(__file__).parent.parent / "config" / "german_signs.json").read_text())["classes"]
    clf = SignClassifier(str(MODELS_DIR / "sign_classifier.pth"), region="german")
    signdet = YOLO(str(MODELS_DIR / "german_sign_detector.engine"), task="detect")

    cap = cv2.VideoCapture(VIDEO)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"\n\nPART B — sign classifier confidence on real detected crops")
    print("GTSRB has 43 classes and NO U-turn sign, so U-turns are out-of-distribution.")
    print("Checking whether top1-vs-top2 margin can flag unreliable labels.\n")

    rows = []
    for i in range(FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        proc = cv2.resize(frame, (int(w * 512 / h), 512))
        sc = 512 / h
        for r in signdet(proc, conf=0.25, verbose=False, imgsz=480):
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v / sc) for v in b.xyxy[0].tolist()]
                if (x2 - x1) < 24 or (y2 - y1) < 24:
                    continue
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                probs = clf.predict_proba(roi) if hasattr(clf, "predict_proba") else None
                if probs is None:
                    res = clf.classify(roi)
                    if res:
                        rows.append((res[1], None, res[0]))
                    continue
                order = np.argsort(probs)[::-1]
                p1, p2 = probs[order[0]], probs[order[1]]
                rows.append((float(p1), float(p1 - p2), names[str(int(order[0]))]))

    cap.release()
    if not rows:
        print("  No sign crops large enough to classify.")
        return

    conf = np.array([r[0] for r in rows])
    print(f"  {len(rows)} classified crops")
    print(f"  confidence: min {conf.min():.2f}  median {np.median(conf):.2f}  max {conf.max():.2f}")
    print(f"  fraction with conf > 0.60 (current gate): {(conf > 0.60).mean()*100:.0f}%")
    print(f"  fraction with conf > 0.90:                {(conf > 0.90).mean()*100:.0f}%")

    margins = [r[1] for r in rows if r[1] is not None]
    if margins:
        m = np.array(margins)
        print(f"  top1-top2 margin: median {np.median(m):.2f}, "
              f"frac > 0.50: {(m > 0.50).mean()*100:.0f}%")

    print("\n  Most frequent labels assigned:")
    for lbl, c in Counter(r[2] for r in rows).most_common(8):
        print(f"    {c:4d}x  {lbl}")


if __name__ == "__main__":
    part_a()
    part_b()
