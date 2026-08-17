"""CarLaneI — Autonomous Vehicle Lane & Traffic Detection using Computer Vision.

Focus: lane detection + traffic signal detection.
Models: YOLOP (lane segmentation + drivable area) + YOLOv8n (vehicles + traffic lights).
OpenCV: lane-curve fitting, morphological refinement, perspective analysis.
CNN: YOLOP backbone (trained, lane/area segmentation).
Framework: PyTorch + TensorRT for inference.

Architecture (2 models, 1 lane fitter):
  YOLOP (TRT)  → drivable area mask + lane line mask
  YOLOv8n (TRT) → vehicle boxes + traffic light boxes + state classification
  LaneFitter (OpenCV) → connected-component lane curves + ego-lane offset + LDW
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR, ensure_dirs
ensure_dirs()


def run_pipeline(input_path, output_path=None, crop_center=True, live=False,
                 max_proc_h=512, display_h=0):
    device = torch.device("cuda")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading models...")

    YOLOP_SZ = 384
    yolop_engine = MODELS_DIR / f"yolop_{YOLOP_SZ}.engine"
    yolop_trt = None
    yolop_pt = None
    if yolop_engine.exists():
        from trt_runner import TRTSeg
        yolop_trt = TRTSeg(yolop_engine, imgsz=YOLOP_SZ)
        print("  YOLOP (TensorRT) ✓")
    else:
        yolop_pt = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True, trust_repo=True)
        yolop_pt.eval().to(device)
        print("  YOLOP (PyTorch) ✓")

    from ultralytics import YOLO
    yolov8_engine = MODELS_DIR / "yolov8n.engine"
    if yolov8_engine.exists():
        yolov8 = YOLO(str(yolov8_engine), task="detect")
        print("  YOLOv8n (TensorRT) ✓")
    else:
        yolov8 = YOLO(str(MODELS_DIR / "yolov8n.pt"))
        print("  YOLOv8n ✓")

    VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- Video setup ---
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if crop_center and orig_w / orig_h > 2.5:
        third = orig_w // 3
        crop_x1, crop_x2 = third, 2 * third
        frame_w = third
    else:
        crop_x1, crop_x2 = 0, orig_w
        frame_w = orig_w
    frame_h = orig_h

    print(f"\nInput: {input_path}")
    print(f"  {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames ({total_frames/fps:.0f}s)")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    if live:
        disp_h = display_h if display_h > 0 else frame_h
        disp_w = int(frame_w * disp_h / frame_h)
        cv2.namedWindow("CarLaneI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CarLaneI", disp_w, disp_h)

    # --- State ---
    cached_vehicles = []
    cached_lights = []
    cached_tl_state = ""
    lane_idx = None

    # Ego corridor from the drivable-area mask. Lane-line pairing was measured to
    # work in only 4 of ~2800 frames, so the DA mask is the reliable signal and
    # lane markings are used to refine the corridor rather than define it.
    from ego_corridor import EgoCorridor
    from traffic_light import TrafficLightTracker
    corridor = EgoCorridor(frame_w, frame_h)
    tl_tracker = TrafficLightTracker(frame_h)
    have_fit = False
    ldw_state = ""

    YOLOP_INTERVAL = 4
    YOLO_INTERVAL = 3

    print(f"  Strategy: YOLOP/{YOLOP_INTERVAL}, YOLOv8/{YOLO_INTERVAL}")
    print(f"\nProcessing...")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, full_frame = cap.read()
        if not ret:
            break

        frame = full_frame[:, crop_x1:crop_x2]
        frame_count += 1
        output = frame.copy()

        # --- YOLOP: lane + drivable area (every Nth frame) ---
        if frame_count % YOLOP_INTERVAL == 1:
            proc = cv2.resize(frame, (int(frame_w * max_proc_h / frame_h), max_proc_h)) \
                   if frame_h > max_proc_h else frame
            img = cv2.cvtColor(cv2.resize(proc, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img).to(device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()

            if yolop_trt is not None:
                da_t, ll_t = yolop_trt.infer(tensor)
                da = da_t.squeeze().cpu().numpy()
                ll = ll_t.squeeze().cpu().numpy()
            else:
                with torch.no_grad():
                    _, da_seg, ll_seg = yolop_pt(tensor)
                da = torch.argmax(da_seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                ll = torch.argmax(ll_seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            da_full = cv2.resize(da, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            ll_full = cv2.resize(ll, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

            lane_thick = cv2.dilate(ll_full, np.ones((3, 3), np.uint8), iterations=2)
            lane_idx = lane_thick == 1

            # Corridor from road surface, snapped to lane markings where present
            have_fit = corridor.update(da_full, lane_mask=lane_thick)
            off = corridor.offset_ratio()
            if off is None:
                ldw_state = ""
            elif abs(off) > 0.80:
                ldw_state = "LANE DEPARTURE"
            elif abs(off) > 0.60:
                ldw_state = "drifting"
            else:
                ldw_state = ""

        # --- YOLOv8: vehicles + traffic lights (every Nth frame) ---
        if frame_count % YOLO_INTERVAL == 1:
            results = yolov8(frame, conf=0.2, verbose=False, imgsz=640)
            cached_vehicles = []
            tl_candidates = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if cls_id == 9:
                        tl_candidates.append((x1, y1, x2, y2, conf))
                    elif cls_id in (2, 3, 5, 7):
                        cached_vehicles.append((x1, y1, x2, y2, VEHICLE_NAMES[cls_id], conf))

            # Geometry gate + temporal state voting. Rejects sub-horizon false
            # positives (53% of raw accepts) and stops colour flipping (11%).
            cached_lights = tl_tracker.update(tl_candidates, frame)
            cached_tl_state = tl_tracker.dominant_state(cached_lights)

        # --- Draw ---
        # Only highlight the EGO LANE (between the two fitted curves), not the
        # entire drivable area. If the fitter doesn't have a confident ego pair,
        # draw nothing — no misleading green.
        # Ego corridor first, so boxes draw on top of it
        if have_fit:
            corridor.draw(output, fill=True)
        if lane_idx is not None:
            # Raw lane markings stay visible as thin highlights
            output[lane_idx] = (170, 255, 120)

        for (x1, y1, x2, y2, name, conf) in cached_vehicles:
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(output, f"{name} {conf:.0%}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 2)

        for (x1, y1, x2, y2, state, conf) in cached_lights:
            color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255),
                     "GREEN": (0, 255, 0)}.get(state, (200, 200, 200))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            if y1 > 20:
                cv2.putText(output, state, (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD
        elapsed = time.time() - start_time
        cur_fps = frame_count / elapsed if elapsed > 0 else 0
        strip = output[0:40]
        strip[:] = (strip * 0.3).astype(np.uint8)
        n_cars = len(cached_vehicles)
        n_lights = len(cached_lights)
        cv2.putText(output, f"CarLaneI | {cur_fps:.0f} FPS | cars:{n_cars} lights:{n_lights}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if ldw_state:
            col = (0, 0, 255) if ldw_state == "LANE DEPARTURE" else (0, 200, 255)
            cv2.putText(output, ldw_state, (frame_w // 2 - 110, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if cached_tl_state:
            color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255),
                     "GREEN": (0, 255, 0)}.get(cached_tl_state, (200, 200, 200))
            cv2.putText(output, cached_tl_state, (frame_w - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        # Output
        if writer:
            writer.write(output)
        if live:
            show = cv2.resize(output, (disp_w, disp_h)) \
                   if display_h > 0 and frame_h > display_h else output
            cv2.imshow("CarLaneI", show)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                print("\n  [User quit]")
                break

        if frame_count % 100 == 0:
            print(f"  {frame_count*100//total_frames}% ({frame_count}/{total_frames}) | FPS: {cur_fps:.1f}")

    cap.release()
    if writer:
        writer.release()
    if live:
        cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"\nDone! {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")
    if output_path:
        print(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CarLaneI — Lane & Traffic Detection")
    p.add_argument("--input", default="data/frankfurt_clip.mp4")
    p.add_argument("--output", default="output/demo.mp4")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--live", action="store_true", help="Show live window")
    p.add_argument("--no-record", action="store_true", help="Skip writing output file")
    p.add_argument("--display-h", type=int, default=0, help="Display height (0=native)")
    args = p.parse_args()

    out = None if args.no_record else args.output
    run_pipeline(args.input, out, crop_center=not args.no_crop,
                 live=args.live, display_h=args.display_h)
