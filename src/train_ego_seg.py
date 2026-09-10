"""Fine-tune YOLOv8-seg to segment the EGO LANE, on real BDD100K labels.

Replaces the hand-tuned OpenCV corridor (ego_corridor.py) with a learned ego-lane
region. The corridor was model-based heuristics on YOLOP's generic drivable mask;
this is a model trained directly on 8k human-labelled ego-lane masks (BDD "direct
drivable"). The measurable claim for judges: learned ego mask vs heuristic corridor
on the same held-out clips (IoU / spill / jitter — see eval_test_set / capture_diagnostics).

yolov8n-seg on 6GB: imgsz 640, batch 8. ~real training, real held-out val.

REQUIRES GPU.  Pre-flight without training:  python src/train_ego_seg.py --check
Train:  python src/train_ego_seg.py
Then:   python src/export_tensorrt.py   (after adding ego_seg — or export inline)
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

DATA_YAML = PROJECT_ROOT / "data" / "ego_seg" / "dataset.yaml"
OUT_MODEL = MODELS_DIR / "ego_seg.pt"


def check_ready():
    if not DATA_YAML.exists():
        print(f"  MISSING {DATA_YAML} — run prepare_ego_seg.py first")
        return False
    import yaml
    cfg = yaml.safe_load(DATA_YAML.read_text())
    root = Path(cfg["path"])
    ok = True
    for split in ("train", "val"):
        imgs = list((root / split / "images").glob("*"))
        lbls = list((root / split / "labels").glob("*.txt"))
        with_poly = sum(1 for l in lbls if l.stat().st_size > 0)
        print(f"  {split}: {len(imgs)} images, {with_poly} with ego polygon")
        if not imgs:
            ok = False
    return ok


def train(epochs=60, imgsz=640, batch=8, device=0, resume=False):
    from ultralytics import YOLO

    # Resume picks up the exact optimizer/epoch/LR state from last.pt.
    last = PROJECT_ROOT / "runs" / "ego_seg" / "weights" / "last.pt"
    if resume:
        if not last.exists():
            print(f"  Can't resume — no checkpoint at {last}")
            return None
        print(f"\n{'='*60}\nRESUMING training from {last}\n{'='*60}")
        model = YOLO(str(last))
        model.train(resume=True)
        best = PROJECT_ROOT / "runs" / "ego_seg" / "weights" / "best.pt"
        if best.exists():
            shutil.copy2(best, OUT_MODEL)
            print(f"\n  Saved: {OUT_MODEL}")
        return OUT_MODEL

    if not check_ready():
        print("\nDataset not ready — aborting before GPU.")
        return None
    print(f"\n{'='*60}\nTRAINING: YOLOv8n-seg ego lane on real BDD masks\n{'='*60}")
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, workers=2, patience=12,
        project=str(PROJECT_ROOT / "runs"), name="ego_seg", exist_ok=True,
        pretrained=True,
        # road scenes: don't vertically flip; mild geometric aug; lighting jitter
        fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.5, hsv_v=0.4, mosaic=1.0,
    )
    best = PROJECT_ROOT / "runs" / "ego_seg" / "weights" / "best.pt"
    if not best.exists():
        best = best.with_name("last.pt")
    if best.exists():
        shutil.copy2(best, OUT_MODEL)
        print(f"\n  Saved: {OUT_MODEL}")
    else:
        print("\n  WARNING: no weights produced.")
    return OUT_MODEL


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="resume from runs/ego_seg/weights/last.pt")
    a = p.parse_args()
    if a.check:
        print("Pre-flight:")
        print("READY" if check_ready() else "NOT READY")
    else:
        train(a.epochs, a.imgsz, a.batch, a.device, resume=a.resume)
