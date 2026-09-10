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
import os
import time
from pathlib import Path

# Quiet OpenCV's FFmpeg backend: partially-corrupt or oddly-encoded source
# videos otherwise flood the console with harmless "Invalid NAL unit size /
# Error splitting the input into NAL units" decoder warnings. Must be set
# BEFORE cv2 is imported. "fatal" keeps genuine failures visible.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # -8 = AV_LOG_QUIET+ (silence)

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR, ensure_dirs
ensure_dirs()


def run_pipeline(input_path, output_path=None, crop_center=True, live=False,
                 max_proc_h=512, display_h=0, telemetry_out=None,
                 progress_cb=None, force_corridor=False, lane_model="auto",
                 frame_cb=None, fast=False, stop_cb=None):
    """Process a video with the full perception stack.

    telemetry_out: optional path to write a JSON of REAL per-frame detections
        and an aggregate summary (for the web showcase). No fabricated values.
    progress_cb: optional callable(pct:int) for a UI progress bar.
    frame_cb: optional callable(annotated_bgr_frame) fired once per processed
        frame — lets the web layer stream a live MJPEG preview while rendering.
    force_corridor: if True, use the heuristic EgoCorridor for lanes even when
        the learned ego-seg model exists (for before/after comparison in the UI).
    """
    import json as _json
    device = torch.device("cuda")
    print(f"Device: {device}")
    telemetry = {"frames": [], "summary": {}} if telemetry_out else None

    # --- Load models ---
    print("Loading models...")

    YOLOP_SZ = 384
    yolop_engine = MODELS_DIR / f"yolop_{YOLOP_SZ}.engine"
    yolop_trt = None
    yolop_pt = None
    if yolop_engine.exists():
        from trt_runner import TRTSeg
        yolop_trt = TRTSeg(yolop_engine, imgsz=YOLOP_SZ)
        print("  YOLOP (TensorRT) [ok]")
    else:
        yolop_pt = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True, trust_repo=True)
        yolop_pt.eval().to(device)
        print("  YOLOP (PyTorch) [ok]")

    from ultralytics import YOLO
    yolov8_engine = MODELS_DIR / "yolov8n.engine"
    if yolov8_engine.exists():
        yolov8 = YOLO(str(yolov8_engine), task="detect")
        print("  YOLOv8n (TensorRT) [ok]")
    else:
        yolov8 = YOLO(str(MODELS_DIR / "yolov8n.pt"))
        print("  YOLOv8n [ok]")

    # Sign detector — single-stage, trained on REAL GTSDB (mAP50 0.914 held-out).
    # 4 super-classes the sign's shape+colour genuinely support; no separate
    # saturated classifier. Optional: pipeline runs fine without it.
    # Prefer the MERGED German+Indian detector (real GTSDB + real Indian dashcam
    # signs) when present, since the showcase footage is Indian; fall back to the
    # German-only detector. Both emit the SAME 4 super-classes, so nothing
    # downstream changes - only which weights back them.
    sign_det = None
    sign_provenance = None
    # (engine, pt, provenance) candidates in preference order
    _sign_candidates = [
        (MODELS_DIR / "signs_merged_detector.engine",
         MODELS_DIR / "signs_merged_detector.pt",
         "YOLOv8n on GTSDB(DE)+Indian dashcam, 4 super-classes"),
        (MODELS_DIR / "german_sign_detector.engine",
         MODELS_DIR / "german_sign_detector.pt",
         "YOLOv8n on GTSDB, 4 classes (mAP50 0.914)"),
    ]
    for _eng, _pt, _prov in _sign_candidates:
        if _eng.exists():
            sign_det = YOLO(str(_eng), task="detect")
            sign_provenance = _prov
            print(f"  Sign detector (TensorRT) [ok] - {_pt.stem}")
            break
        if _pt.exists():
            sign_det = YOLO(str(_pt))
            sign_provenance = _prov
            print(f"  Sign detector (PyTorch) [ok] - {_pt.stem}")
            break
    if sign_det is None:
        print("  Sign detector: not found, skipping signs")
    SIGN_NAMES = {0: "prohibitory", 1: "mandatory", 2: "danger", 3: "other"}
    SIGN_COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 140, 255), 3: (0, 200, 200)}

    # Dedicated traffic-light STATE detector (trained on Bosch: red/green/yellow/off).
    # When present it REPLACES the COCO-box + pixel-colour-heuristic path: it detects
    # the light AND its colour directly, then feeds the SAME temporal voting. Falls
    # back to the heuristic (via YOLOv8 COCO class 9) when this model is absent.
    light_det = None
    light_provenance = None
    _light_eng = MODELS_DIR / "light_state.engine"
    _light_pt = MODELS_DIR / "light_state.pt"
    LIGHT_NAMES = {0: "RED", 1: "GREEN", 2: "YELLOW", 3: "OFF"}
    LIGHT_CONF = 0.12   # calibrated to Indian footage (see per-frame use below)
    if _light_eng.exists():
        light_det = YOLO(str(_light_eng), task="detect")
        light_provenance = "YOLOv8n light-state detector (Bosch: red/green/yellow/off)"
        print("  Light-state detector (TensorRT) [ok]")
    elif _light_pt.exists():
        light_det = YOLO(str(_light_pt))
        light_provenance = "YOLOv8n light-state detector (Bosch: red/green/yellow/off)"
        print("  Light-state detector (PyTorch) [ok]")
    else:
        print("  Light-state detector: not found, using YOLOP+heuristic colour")

    VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- Video setup ---
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS)) or 30
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
    cached_signs = []
    lane_idx = None

    # Ego corridor from the drivable-area mask. Lane-line pairing was measured to
    # work in only 4 of ~2800 frames, so the DA mask is the reliable signal and
    # lane markings are used to refine the corridor rather than define it.
    from ego_corridor import EgoCorridor
    from traffic_light import TrafficLightTracker
    corridor = EgoCorridor(frame_w, frame_h)
    tl_tracker = TrafficLightTracker(frame_h)

    # Learned ego-lane segmenter (trained on 8k real BDD ego-lane masks). Used as
    # the lane source when its model is present; the heuristic corridor above is
    # the fallback. This is the measured lane upgrade over the DA-mask corridor.
    ego_seg = None
    # Which lane weights to use:
    #   'auto'        -> BDD ego_seg (default, mask mAP50 0.972)
    #   'ego_seg'     -> force BDD (US) model
    #   'ego_seg_idd' -> force IDD (Indian) model (weak: 0.14 mAP50, A/B only)
    def _lane_choice():
        def _has(name):
            return (MODELS_DIR / f"{name}.engine").exists() or (MODELS_DIR / f"{name}.pt").exists()
        if lane_model in ("ego_seg", "ego_seg_idd"):
            return lane_model if _has(lane_model) else None
        # auto: BDD ego_seg is the default (mask mAP50 0.972). The IDD model
        # measured only 0.14 mask mAP50, so 'auto' never picks it — it stays
        # selectable explicitly (ego_seg_idd) for A/B only.
        # ponytail: hard-preference, not footage-aware — fine until IDD is retrained to beat BDD.
        if _has("ego_seg"):
            return "ego_seg"
        if _has("ego_seg_idd"):
            return "ego_seg_idd"
        return None

    if force_corridor:
        print("  Lane source: heuristic EgoCorridor (forced, before/after mode)")
    else:
        _chosen = _lane_choice()
        if _chosen:
            try:
                from ego_seg_runner import EgoSegRunner
                ego_seg = EgoSegRunner(model_name=_chosen)
                print(f"  Ego-lane segmenter ({ego_seg.kind}, {_chosen}) [ok]")
            except Exception as e:
                print(f"  Ego-lane segmenter unavailable ({e}); using corridor")
    have_fit = False
    ldw_state = ""

    # fast (live-preview) mode widens the skip cadence. Detections persist in
    # the caches between runs, so the overlay stays continuous — only the
    # refresh rate of boxes drops, which is invisible at 25+ display FPS. The
    # default (recorded/web) cadence is unchanged so telemetry stays accurate.
    # ponytail: fixed intervals, not adaptive to measured frame time. Ceiling —
    # on a slower GPU 6/4 may still dip; upgrade path = target-FPS auto-tune.
    YOLOP_INTERVAL = 6 if fast else 4
    YOLO_INTERVAL = 4 if fast else 3

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

            # Lane: learned ego-seg model if available, else heuristic corridor.
            if ego_seg is not None:
                have_fit = ego_seg.update(frame)
                off = ego_seg.offset_ratio()
            else:
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

            # Lights: prefer the trained state detector (detects box + colour
            # directly); else fall back to COCO boxes + pixel-colour heuristic.
            # Both feed the SAME temporal voting so colour can't flip per frame.
            if light_det is not None:
                stated = []
                # ponytail: 0.15 calibrated to real Indian footage — the Bosch-
                # trained model scores signals here at ~0.12-0.20 (domain gap),
                # far below its 0.30 native conf. Temporal voting in
                # update_stated() rejects one-frame flukes, so a low raw gate is
                # safe. Ceiling: a light retrained on Indian signals would fire
                # at higher conf and let us raise this back.
                for r in light_det(frame, conf=LIGHT_CONF, verbose=False, imgsz=640):
                    for box in r.boxes:
                        st = LIGHT_NAMES.get(int(box.cls[0]), "OFF")
                        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
                        stated.append((lx1, ly1, lx2, ly2, st, float(box.conf[0])))
                cached_lights = tl_tracker.update_stated(stated, frame)
            else:
                # Geometry gate + temporal state voting. Rejects sub-horizon false
                # positives (53% of raw accepts) and stops colour flipping (11%).
                cached_lights = tl_tracker.update(tl_candidates, frame)
            cached_tl_state = tl_tracker.dominant_state(cached_lights)

            # Signs — single-stage, real-data detector. conf 0.35: the detector
            # earns its confidence on real GTSDB, unlike the old saturated
            # classifier, so a real threshold (not a margin gate) is enough.
            if sign_det is not None:
                cached_signs = []
                for r in sign_det(frame, conf=0.35, verbose=False, imgsz=640):
                    for box in r.boxes:
                        sc = int(box.cls[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cached_signs.append((x1, y1, x2, y2, sc, float(box.conf[0])))

        # --- Draw ---
        # Only highlight the EGO LANE (between the two fitted curves), not the
        # entire drivable area. If the fitter doesn't have a confident ego pair,
        # draw nothing — no misleading green.
        # Ego corridor first, so boxes draw on top of it
        if have_fit:
            if ego_seg is not None:
                ego_seg.draw(output)
            else:
                corridor.draw(output, fill=True)
        if lane_idx is not None:
            # Raw YOLOP lane markings stay visible as thin highlights
            output[lane_idx] = (170, 255, 120)

        for (x1, y1, x2, y2, name, conf) in cached_vehicles:
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(output, f"{name} {conf:.0%}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 2)

        for (x1, y1, x2, y2, sc, conf) in cached_signs:
            col = SIGN_COLORS.get(sc, (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), col, 2)
            cv2.putText(output, f"{SIGN_NAMES.get(sc, '?')} {conf:.0%}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)

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
        n_signs = len(cached_signs)
        cv2.putText(output,
                    f"CarLaneI | {cur_fps:.0f} FPS | cars:{n_cars} lights:{n_lights} signs:{n_signs}",
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

        # --- Real per-frame telemetry (for the web showcase) ---
        if telemetry is not None:
            telemetry["frames"].append({
                "frame": frame_count,
                "t": round(frame_count / fps, 3),
                "fps": round(cur_fps, 1),
                "lane_offset": None if off is None else round(float(off), 3),
                "ldw": ldw_state,
                "dominant_light": cached_tl_state or None,
                "vehicles": [
                    {"box": [x1, y1, x2, y2], "cls": name, "conf": round(conf, 3)}
                    for (x1, y1, x2, y2, name, conf) in cached_vehicles],
                "signs": [
                    {"box": [x1, y1, x2, y2], "cls": SIGN_NAMES.get(sc, "?"),
                     "conf": round(conf, 3)}
                    for (x1, y1, x2, y2, sc, conf) in cached_signs],
                "lights": [
                    {"box": [x1, y1, x2, y2], "state": st, "conf": round(cf, 3)}
                    for (x1, y1, x2, y2, st, cf) in cached_lights],
            })
        if progress_cb and total_frames and frame_count % 15 == 0:
            progress_cb(min(99, frame_count * 100 // total_frames))

        # Output
        if writer:
            writer.write(output)
        if frame_cb is not None:
            frame_cb(output)
        # Cooperative cancellation (web long-live: stop when client leaves or a
        # new run starts) — checked every frame so it stops promptly.
        if stop_cb is not None and stop_cb():
            print("\n  [Cancelled]")
            break
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
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nDone! {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    if output_path:
        print(f"Output: {output_path}")

    # --- Aggregate real telemetry summary + write JSON ---
    if telemetry is not None:
        fr = telemetry["frames"]
        offs = [f["lane_offset"] for f in fr if f["lane_offset"] is not None]
        lane_frames = sum(1 for f in fr if f["lane_offset"] is not None)
        ldw_frames = sum(1 for f in fr if f["ldw"])
        # unique sign classes seen and light-state exposure
        sign_counts, light_counts = {}, {}
        veh_total = 0
        for f in fr:
            veh_total += len(f["vehicles"])
            for s in f["signs"]:
                sign_counts[s["cls"]] = sign_counts.get(s["cls"], 0) + 1
            for l in f["lights"]:
                light_counts[l["state"]] = light_counts.get(l["state"], 0) + 1
        telemetry["summary"] = {
            "source": str(input_path),
            "output_video": str(output_path) if output_path else None,
            "frames": frame_count,
            "duration_s": round(frame_count / fps, 1) if fps else None,
            "fps_source": fps,
            "avg_processing_fps": round(avg_fps, 1),
            "resolution": [frame_w, frame_h],
            "lane_present_pct": round(lane_frames / max(frame_count, 1) * 100, 1),
            "ldw_events_pct": round(ldw_frames / max(frame_count, 1) * 100, 1),
            "mean_abs_lane_offset": round(float(np.mean(np.abs(offs))), 3) if offs else None,
            "vehicles_per_frame": round(veh_total / max(frame_count, 1), 2),
            "sign_class_counts": sign_counts,
            "light_state_counts": light_counts,
            # provenance: real, measured model metrics (not fabricated)
            "lane_method": "corridor" if (ego_seg is None) else "egoseg",
            "models": {
                "ego_lane": ("heuristic EgoCorridor (YOLOP drivable mask)"
                             if ego_seg is None else
                             ("YOLOv8n-seg on IDD ego-lane (Indian roads)"
                              if getattr(ego_seg, "model_name", "") == "ego_seg_idd"
                              else "YOLOv8n-seg on BDD100K ego-lane (mask mAP50 0.972)")),
                "signs": (sign_provenance if sign_provenance
                          else "sign detector not loaded"),
                "drivable": "YOLOP (BDD100K)",
                "vehicles": "YOLOv8n COCO",
                "lights": (light_provenance if light_provenance
                           else "YOLOv8n COCO box + colour heuristic + temporal voting"),
            },
        }
        Path(telemetry_out).parent.mkdir(parents=True, exist_ok=True)
        with open(telemetry_out, "w") as tf:
            _json.dump(telemetry, tf)
        print(f"Telemetry: {telemetry_out}")
    if progress_cb:
        progress_cb(100)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CarLaneI — Lane & Traffic Detection")
    p.add_argument("--input", default="data/frankfurt_clip.mp4")
    p.add_argument("--output", default="output/demo.mp4")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--live", action="store_true", help="Show live window")
    p.add_argument("--no-record", action="store_true", help="Skip writing output file")
    p.add_argument("--display-h", type=int, default=0, help="Display height (0=native)")
    p.add_argument("--fast", action="store_true",
                   help="Live-preview cadence (wider skip) for higher display FPS")
    args = p.parse_args()

    out = None if args.no_record else args.output
    run_pipeline(args.input, out, crop_center=not args.no_crop,
                 live=args.live, display_h=args.display_h, fast=args.fast)
