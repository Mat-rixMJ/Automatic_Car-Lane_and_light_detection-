"""Self-analysis: measure what the current pipeline actually detects and misses."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR
from trt_runner import TRTSeg
from ultralytics import YOLO
from lane_fit import LaneFitter

YOLOP_SZ = 384
device = torch.device("cuda")
yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
yolov8 = YOLO(str(MODELS_DIR / "yolov8n.engine"), task="detect")

videos = [
    ("Frankfurt 720p (60s)", "D:/carLane/downloads/frankfurt_720p_5min.mp4", 1770),
    ("BDDA test/100", "D:/carLane/BDDA/test/camera_videos/100.mp4", 600),
    ("BDDA test/1003", "D:/carLane/BDDA/test/camera_videos/1003.mp4", 420),
    ("BDDA test/1045", "D:/carLane/BDDA/test/camera_videos/1045.mp4", 600),
    ("BDDA test/1072", "D:/carLane/BDDA/test/camera_videos/1072.mp4", 600),
]

for label, path, max_f in videos:
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fitter = LaneFitter(w, h)
    n = tl = veh = lane_fit = ego_pair = 0
    states = {"RED": 0, "GREEN": 0, "YELLOW": 0, "UNK": 0}
    for i in range(max_f):
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
        _, ll_t = yolop.infer(t)
        ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        ok = fitter.update(ll)
        if ok:
            lane_fit += 1
        if fitter.offset_ratio() is not None:
            ego_pair += 1

        for r in yolov8(frame, conf=0.2, verbose=False, imgsz=640):
            for box in r.boxes:
                cid = int(box.cls[0])
                if cid == 9:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bh, bw = y2 - y1, x2 - x1
                    if bh >= 12 and (bw / bh if bh else 1) < 1.0 and y1 < h * 0.75:
                        tl += 1
                        roi = frame[y1:y2, x1:x2]
                        th2 = max(1, roi.shape[0] // 3)
                        top_r = float(roi[:th2, :, 2].mean()) - float(roi[:th2, :, 1].mean())
                        bot_g = float(roi[2*th2:, :, 1].mean()) - float(roi[2*th2:, :, 2].mean())
                        mid_y = float(roi[th2:2*th2, :, 1].mean()) + float(roi[th2:2*th2, :, 2].mean())
                        if bot_g > 8:
                            states["GREEN"] += 1
                        elif top_r > 8:
                            states["RED"] += 1
                        elif mid_y > (top_r + bot_g) + 120:
                            states["YELLOW"] += 1
                        else:
                            states["UNK"] += 1
                elif cid in (2, 3, 5, 7):
                    veh += 1
    cap.release()
    classified = tl - states["UNK"]
    print(f"\n{label} ({n} frames, {w}x{h})")
    print(f"  Lane fit: {lane_fit/n*100:.0f}% | Ego pair: {ego_pair/n*100:.0f}%")
    print(f"  Vehicles: {veh} ({veh/n:.1f}/frame)")
    print(f"  Traffic lights: {tl} ({tl/n:.2f}/frame)")
    print(f"  TL state: R={states['RED']} G={states['GREEN']} Y={states['YELLOW']} ?={states['UNK']}")
    if tl:
        print(f"  State classified: {classified/tl*100:.0f}%")
