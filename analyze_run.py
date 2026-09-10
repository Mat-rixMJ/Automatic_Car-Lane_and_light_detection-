"""Run the pipeline on an Indian clip and print a per-class telemetry breakdown.

Usage: python analyze_run.py <clip> <seconds>
Prints lane presence, signs-by-class, lights-by-state, vehicles/frame — the
numbers we tune against. No fabricated values; all from the real run.
"""
import sys, json, cv2
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import run_pipeline_fast as rp

clip = sys.argv[1] if len(sys.argv) > 1 else "downloads/kolkata_720p_seg.mp4"
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# Trim to `seconds` for a quick, representative pass.
src = Path(clip)
trimmed = Path("output") / "web" / f"_analyze_src.mp4"
trimmed.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
n = int(fps * seconds); w = int(cap.get(3)); h = int(cap.get(4))
vw = cv2.VideoWriter(str(trimmed), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
while i < n:
    ret, fr = cap.read()
    if not ret: break
    vw.write(fr); i += 1
cap.release(); vw.release()
print(f"clip={src.name}  fps={fps:.1f}  trimmed_frames={i}  ({seconds}s)")

out_json = Path("output") / "web" / "_analyze.json"
rp.run_pipeline(str(trimmed), str(Path("output") / "web" / "_analyze.mp4"),
                crop_center=False, telemetry_out=str(out_json))

data = json.load(open(out_json))
s = data["summary"]; frames = data["frames"]
print("\n===== SUMMARY =====")
for k, v in s.items():
    print(f"  {k}: {v}")

# Per-frame aggregates (keys per telemetry schema in run_pipeline_fast.py)
lane_present = sum(1 for f in frames if f.get("lane_offset") is not None)
dom_light = Counter()          # voted per-frame dominant light state
light_det_cls = Counter()      # raw per-detection light states
sign_cls = Counter()
veh = 0
sign_confs = []
light_confs = []
for f in frames:
    for sg in f.get("signs", []):
        sign_cls[sg["cls"]] += 1
        sign_confs.append(sg["conf"])
    dl = f.get("dominant_light")
    if dl: dom_light[dl] += 1
    for lt in f.get("lights", []):
        light_det_cls[lt["state"]] += 1
        light_confs.append(lt["conf"])
    veh += len(f.get("vehicles", []))

nf = len(frames)
def _stats(xs):
    if not xs: return "none"
    xs = sorted(xs)
    return f"min={xs[0]:.2f} med={xs[len(xs)//2]:.2f} max={xs[-1]:.2f} n={len(xs)}"

print(f"\n===== PER-FRAME ({nf} frames) =====")
print(f"  lane_present (offset!=None): {lane_present}/{nf} ({100*lane_present/max(nf,1):.0f}%)")
print(f"  vehicles total: {veh}  ({veh/max(nf,1):.2f}/frame)")
print(f"  signs by class: {dict(sign_cls)}")
print(f"  sign conf: {_stats(sign_confs)}")
print(f"  lights (voted dominant/frame): {dict(dom_light)}")
print(f"  lights (raw detections by state): {dict(light_det_cls)}")
print(f"  light conf: {_stats(light_confs)}")
