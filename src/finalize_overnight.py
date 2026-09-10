"""Run AFTER ego_seg training completes: lock in the final model + results.

Waits for training to finish (best.pt stops changing), then:
  1. copies runs/ego_seg/weights/best.pt -> models/ego_seg.pt
  2. re-exports the TensorRT engine from the FINAL weights
  3. re-runs the lane IoU eval on held-out BDD val
  4. renders a fresh demo video with the final, fast model
  5. writes output/FINAL_RESULTS.txt with the numbers

Designed to be launched once and left overnight alongside training.

  python src/finalize_overnight.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

PY = sys.executable
BEST = PROJECT_ROOT / "runs" / "ego_seg" / "weights" / "best.pt"
EGO_PT = MODELS_DIR / "ego_seg.pt"
EGO_ENGINE = MODELS_DIR / "ego_seg.engine"
DEMO_CLIP = PROJECT_ROOT / "BDDA" / "test" / "camera_videos" / "1003.mp4"
OUT = PROJECT_ROOT / "output"


def wait_for_training_done(poll=120, stable_for=600):
    """Block until best.pt hasn't changed for `stable_for` seconds.

    Training rewrites best.pt whenever val improves and last.pt every epoch;
    once both stop moving for 10 min, training has ended (or early-stopped).
    """
    print("Waiting for training to finish (watching best.pt)...")
    last_mtime = None
    stable_since = None
    while True:
        if BEST.exists():
            m = BEST.stat().st_mtime
            if m == last_mtime:
                if stable_since and (time.time() - stable_since) >= stable_for:
                    print("  best.pt stable -> training done.")
                    return
            else:
                last_mtime = m
                stable_since = time.time()
                print(f"  best.pt updated at {time.strftime('%H:%M:%S')}; still training")
        time.sleep(poll)


def sh(cmd, timeout=1800):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((r.stdout or "").splitlines()[-15:])
    print(tail)
    if r.returncode != 0:
        print("  STDERR tail:", "\n".join((r.stderr or "").splitlines()[-8:]))
    return r.stdout or ""


def main():
    wait_for_training_done()

    # 1. publish final weights
    if BEST.exists():
        shutil.copy2(BEST, EGO_PT)
        print(f"Copied {BEST} -> {EGO_PT}")

    # 2. re-export engine from FINAL weights (delete stale first)
    if EGO_ENGINE.exists():
        EGO_ENGINE.unlink()
    EGO_ENGINE.with_suffix(".onnx").unlink(missing_ok=True)
    sh([PY, "-c",
        "from ultralytics import YOLO; "
        f"YOLO(r'{EGO_PT}').export(format='engine', imgsz=640, half=True, device=0)"])

    # 3. final lane IoU
    iou_out = sh([PY, str(PROJECT_ROOT / "src" / "eval_lane_iou.py"), "--n", "300"])

    # 4. fresh demo with the final fast model
    sh([PY, "-c",
        "import sys; sys.path.insert(0, r'" + str(PROJECT_ROOT / "src") + "'); "
        "import run_pipeline_fast as rp; "
        f"rp.run_pipeline(r'{DEMO_CLIP}', r'{OUT / 'demo_final.mp4'}', crop_center=False)"])

    # 5. dump the numbers where they're easy to find in the morning
    OUT.mkdir(exist_ok=True)
    (OUT / "FINAL_RESULTS.txt").write_text(
        "CarLaneI overnight finalize\n"
        f"finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "=== FINAL LANE IoU (held-out BDD val) ===\n"
        + "\n".join(l for l in iou_out.splitlines() if "IoU" in l or "method"
                    in l or "ego-seg" in l or "EgoCorridor" in l or "gain" in l)
        + f"\n\nFinal model: {EGO_PT}\nEngine: {EGO_ENGINE}\n"
        "Demo video: output/demo_final.mp4\n"
        "See RESULTS.md for the full write-up.\n")
    print(f"\nDONE. Summary in {OUT / 'FINAL_RESULTS.txt'}")


if __name__ == "__main__":
    main()
