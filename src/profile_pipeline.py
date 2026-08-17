"""Measure where the pipeline actually spends its time, per sub-stage.

Splits YOLOP into pre / infer / post so we can see whether the cost is the
network or the CPU mask handling around it.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR

VIDEO = r"D:\carLane\downloads\frankfurt_720p_5min.mp4"
N = 120
PROC_H = 512
YOLOP_SZ = 384


def main():
    device = torch.device("cuda")

    from trt_runner import TRTSeg
    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)

    from ultralytics import YOLO
    yolov8 = YOLO(str(MODELS_DIR / "yolov8n.engine"), task="detect")
    signdet = YOLO(str(MODELS_DIR / "german_sign_detector.engine"), task="detect")

    cap = cv2.VideoCapture(VIDEO)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"D:\carLane\output\_profile_tmp.mp4", fourcc, 30, (w, h))

    keys = ("read", "downscale", "yolop_pre", "yolop_infer", "yolop_post",
            "yolov8", "sign", "draw", "write", "imshow")
    t = {k: 0.0 for k in keys}

    ret, f0 = cap.read()
    for _ in range(3):
        yolov8(f0, verbose=False, imgsz=384)
        signdet(f0, verbose=False, imgsz=480)

    road_idx = lane_idx = None

    for _ in range(N):
        s = time.perf_counter(); ret, frame = cap.read(); t["read"] += time.perf_counter() - s
        if not ret:
            break

        s = time.perf_counter()
        scale = PROC_H / h
        proc = cv2.resize(frame, (int(w * scale), PROC_H))
        t["downscale"] += time.perf_counter() - s

        s = time.perf_counter()
        img = cv2.resize(proc, (YOLOP_SZ, YOLOP_SZ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).to(device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
        torch.cuda.synchronize()
        t["yolop_pre"] += time.perf_counter() - s

        s = time.perf_counter()
        da_t, ll_t = yolop.infer(tensor)
        t["yolop_infer"] += time.perf_counter() - s

        s = time.perf_counter()
        da = da_t.squeeze().cpu().numpy()
        ll = ll_t.squeeze().cpu().numpy()
        da_full = cv2.resize(da, (w, h), interpolation=cv2.INTER_NEAREST)
        ll_full = cv2.resize(ll, (w, h), interpolation=cv2.INTER_NEAREST)
        lane_thick = cv2.dilate(ll_full, np.ones((3, 3), np.uint8), iterations=2)
        lane_idx = lane_thick == 1
        road_idx = (da_full == 1) & ~lane_idx
        t["yolop_post"] += time.perf_counter() - s

        s = time.perf_counter(); yolov8(proc, conf=0.3, verbose=False, imgsz=384); t["yolov8"] += time.perf_counter() - s
        s = time.perf_counter(); signdet(proc, conf=0.25, verbose=False, imgsz=480); t["sign"] += time.perf_counter() - s

        s = time.perf_counter()
        output = frame.copy()
        px = output[road_idx]
        if px.size:
            px[:, 1] = np.minimum(255, px[:, 1] * 0.75 + 70)
            output[road_idx] = px
        output[lane_idx] = (255, 255, 0)
        t["draw"] += time.perf_counter() - s

        s = time.perf_counter(); writer.write(output); t["write"] += time.perf_counter() - s
        s = time.perf_counter(); cv2.imshow("p", output); cv2.waitKey(1); t["imshow"] += time.perf_counter() - s

    cap.release(); writer.release(); cv2.destroyAllWindows()
    Path(r"D:\carLane\output\_profile_tmp.mp4").unlink(missing_ok=True)

    print(f"\nPer-frame cost over {N} frames, source {w}x{h}, proc {PROC_H}p:")
    total = sum(t.values())
    for k in sorted(keys, key=lambda k: -t[k]):
        print(f"  {k:12s} {t[k]/N*1000:7.2f} ms   ({t[k]/total*100:4.1f}%)")
    print(f"  {'TOTAL':12s} {total/N*1000:7.2f} ms  -> {N/total:.1f} FPS unskipped")

    # What we'd get with the real skip intervals
    per = (t["read"] + t["downscale"] + t["draw"] + t["write"] + t["imshow"]) / N
    yolop_c = (t["yolop_pre"] + t["yolop_infer"] + t["yolop_post"]) / N
    det_c = (t["yolov8"] + t["sign"]) / N
    for yi, di in ((3, 2), (4, 3), (5, 3)):
        est = per + yolop_c / yi + det_c / di
        print(f"  with YOLOP/{yi}, Det/{di}: {est*1000:.1f} ms -> {1/est:.1f} FPS")


if __name__ == "__main__":
    main()
