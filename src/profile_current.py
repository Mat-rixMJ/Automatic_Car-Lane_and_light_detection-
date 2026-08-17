"""Profile the current pipeline to see if hardware limit is hit."""
import sys, time
sys.path.insert(0, 'D:/carLane/src')
import cv2, numpy as np, torch
from utils import MODELS_DIR
from trt_runner import TRTSeg
from ultralytics import YOLO
from lane_fit import LaneFitter

YOLOP_SZ = 384
device = torch.device('cuda')
yolop = TRTSeg(MODELS_DIR / f'yolop_{YOLOP_SZ}.engine', imgsz=YOLOP_SZ)
yolov8 = YOLO(str(MODELS_DIR / 'yolov8n.engine'), task='detect')
cap = cv2.VideoCapture('D:/carLane/downloads/frankfurt_720p_5min.mp4')
w, h = int(cap.get(3)), int(cap.get(4))
fitter = LaneFitter(w, h)
N = 200
t = {'read': 0, 'yolop': 0, 'yolov8': 0, 'lane_fit': 0, 'draw': 0, 'imshow': 0}

for _ in range(3):
    ret, f = cap.read()
    yolov8(f, verbose=False, imgsz=640)

for i in range(N):
    s = time.perf_counter(); ret, frame = cap.read(); t['read'] += time.perf_counter() - s
    if not ret:
        break

    s = time.perf_counter()
    img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
    tn = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
    da_t, ll_t = yolop.infer(tn)
    da = cv2.resize(da_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
    ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
    t['yolop'] += time.perf_counter() - s

    s = time.perf_counter()
    fitter.update(ll)
    fitter.use_da_fallback(da)
    t['lane_fit'] += time.perf_counter() - s

    s = time.perf_counter()
    yolov8(frame, conf=0.2, verbose=False, imgsz=640)
    t['yolov8'] += time.perf_counter() - s

    s = time.perf_counter()
    out = frame.copy()
    if fitter.left and fitter.right:
        fitter.draw(out, fill=True)
    t['draw'] += time.perf_counter() - s

    s = time.perf_counter()
    cv2.imshow('x', out)
    cv2.waitKey(1)
    t['imshow'] += time.perf_counter() - s

cap.release()
cv2.destroyAllWindows()
total = sum(t.values())
print(f"Per-frame (every stage, no skipping), {N} frames:")
for k in sorted(t, key=lambda k: -t[k]):
    print(f"  {k:10s} {t[k]/N*1000:6.1f} ms  ({t[k]/total*100:4.1f}%)")
print(f"  TOTAL      {total/N*1000:6.1f} ms = {N/total:.0f} FPS unskipped")
print()
per = t['read'] / N + t['draw'] / N + t['imshow'] / N
yo = t['yolop'] / N
det = t['yolov8'] / N
lf = t['lane_fit'] / N
for yi, di in [(4, 3), (5, 4), (6, 4), (6, 5)]:
    est = per + yo / yi + det / di + lf / yi
    print(f"  YOLOP/{yi} Det/{di}: {est*1000:.1f}ms = {1/est:.0f} FPS")
