"""Day vs Night evaluation — the field's known hard case, measured honestly.

The literature repeatedly flags night/low-light as the failure mode for lane,
sign, and traffic-light perception. This runs the real detection stack on a set
of NIGHT clips and a set of DAY clips (selected by measured brightness from
select_test_videos' manifest) and reports how each component holds up.

Metrics per clip-set (mean over clips, detection on every Nth frame):
  lane_present_pct   frames with an ego-lane fix
  vehicles/frame     mean vehicle detections
  lights/frame       mean traffic-light detections (recall proxy)
  mean_conf          mean detection confidence (drops in the dark)

Honest by construction: if night is worse, the numbers say so. No tuning to hide it.

  python src/eval_night.py --clips 6 --seconds 6
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

BDDA = PROJECT_ROOT / "BDDA" / "test" / "camera_videos"
MANIFEST = PROJECT_ROOT / "output" / "test_video_manifest.json"
YOLOP_SZ = 384
VEHICLE_CLS = {2, 3, 5, 7}
LIGHT_CLS = 9


def pick_clips(n):
    """Return (night_clips, day_clips) as lists of paths, by measured brightness."""
    prof = json.load(open(MANIFEST))["all_profiled"]
    prof = [c for c in prof if Path(c["path"]).exists()]
    night = sorted([c for c in prof if c["brightness"] < 70], key=lambda c: c["brightness"])
    day = sorted([c for c in prof if c["brightness"] > 110], key=lambda c: -c["brightness"])
    return ([Path(c["path"]) for c in night[:n]],
            [Path(c["path"]) for c in day[:n]])


def eval_set(paths, seconds, yolop, yolov8, corridor_cls):
    device = torch.device("cuda")
    lane_ok = lane_tot = 0
    veh_pf, light_pf, confs = [], [], []
    for vid in paths:
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        corridor = corridor_cls(w, h)
        n = int(seconds * fps)
        i = 0
        while i < n:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % 3 != 1:          # sample every 3rd frame
                continue
            # lane (drivable corridor availability — model-agnostic baseline)
            img = cv2.cvtColor(cv2.resize(frame, (YOLOP_SZ, YOLOP_SZ)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255).contiguous()
            da_t, _ = yolop.infer(t)
            da = cv2.resize(da_t.squeeze().cpu().numpy().astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST)
            lane_tot += 1
            if corridor.update(da):
                lane_ok += 1
            # vehicles + lights + confidence
            v = l = 0
            for r in yolov8(frame, conf=0.25, verbose=False, imgsz=640):
                for b in r.boxes:
                    c = int(b.cls[0]); cf = float(b.conf[0])
                    if c in VEHICLE_CLS:
                        v += 1; confs.append(cf)
                    elif c == LIGHT_CLS:
                        l += 1; confs.append(cf)
            veh_pf.append(v); light_pf.append(l)
        cap.release()
    return {
        "clips": len(paths),
        "lane_present_pct": round(lane_ok / max(lane_tot, 1) * 100, 1),
        "vehicles_per_frame": round(float(np.mean(veh_pf)) if veh_pf else 0, 2),
        "lights_per_frame": round(float(np.mean(light_pf)) if light_pf else 0, 2),
        "mean_conf": round(float(np.mean(confs)) if confs else 0, 3),
    }


def run(clips, seconds):
    from trt_runner import TRTSeg
    from ultralytics import YOLO
    from ego_corridor import EgoCorridor

    yolop = TRTSeg(MODELS_DIR / f"yolop_{YOLOP_SZ}.engine", imgsz=YOLOP_SZ)
    yolov8 = YOLO(str(MODELS_DIR / "yolov8n.engine"), task="detect")

    night, day = pick_clips(clips)
    print(f"night clips: {[p.name for p in night]}")
    print(f"day clips:   {[p.name for p in day]}\n")
    night_m = eval_set(night, seconds, yolop, yolov8, EgoCorridor)
    day_m = eval_set(day, seconds, yolop, yolov8, EgoCorridor)

    print(f"{'metric':22s} {'DAY':>10s} {'NIGHT':>10s} {'delta':>10s}")
    print("-" * 54)
    for k in ("lane_present_pct", "vehicles_per_frame", "lights_per_frame", "mean_conf"):
        d, n = day_m[k], night_m[k]
        print(f"{k:22s} {d:>10} {n:>10} {round(n - d, 3):>10}")
    out = {"day": day_m, "night": night_m}
    (PROJECT_ROOT / "output" / "night_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved output/night_eval.json")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seconds", type=int, default=6)
    a = p.parse_args()
    run(a.clips, a.seconds)
