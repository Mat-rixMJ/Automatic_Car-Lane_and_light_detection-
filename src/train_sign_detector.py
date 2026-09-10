"""Train a single-stage traffic-sign detector on REAL GTSDB data.

Why this replaces train_combined_detector.py + sign_classifier.py:

  The old design was two stages — a YOLOv8 detector (trained on GTSRB crops pasted
  onto road frames) feeding a 43-class GTSRB CNN classifier. The README documents
  why both failed: the detector's 99.3% mAP was against SYNTHETIC data and only
  ~57% real-world presence, and the classifier was SATURATED (median confidence
  1.00 on out-of-distribution signs — it labelled a real U-turn "Ahead only" at
  full confidence and no gate could filter that).

  This is one stage. The detector emits the class directly, trained on real GTSDB
  road photographs (data/gtsdb_yolo, from prepare_gtsdb_yolo.py). No separate
  classifier means no saturated second stage to over-claim specific names. With 4
  super-classes the box label is something shape+colour genuinely supports, so a
  reported "prohibitory" is trustworthy in a way "Speed limit (30km/h)" was not.

This starts from COCO-pretrained yolov8n.pt, so the backbone already knows road
scenes. Output goes to models/german_sign_detector.pt — the exact name
export_tensorrt.py already exports and the (future) pipeline wiring expects.

REQUIRES A GPU. This session has none; run on the RTX 3050 machine.

  python src/prepare_gtsdb_yolo.py          # once, builds data/gtsdb_yolo
  python src/train_sign_detector.py         # trains, ~20-40 min on RTX 3050
  python src/export_tensorrt.py             # -> german_sign_detector.engine

Dry-run the config without a GPU / without training:
  python src/train_sign_detector.py --check
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

DATA_YAML = PROJECT_ROOT / "data" / "gtsdb_yolo" / "dataset.yaml"
OUT_MODEL = MODELS_DIR / "german_sign_detector.pt"


def check_ready(data_yaml=DATA_YAML):
    """Verify the dataset exists and report what training will consume.

    Runnable without a GPU — this is the pre-flight so a missing dataset fails
    loudly here instead of 30 seconds into a GPU job.
    """
    ok = True
    if not data_yaml.exists():
        print(f"  MISSING dataset.yaml: {data_yaml}")
        print("  Run: python src/prepare_gtsdb_yolo.py   (German only)")
        print("   or: python src/prepare_indian_signs.py  (merged German+Indian)")
        return False

    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    print(f"  dataset.yaml : {data_yaml}")
    print(f"  classes ({cfg['nc']}): {list(cfg['names'].values())}")
    for split in ("train", "val"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        flag = "" if n_img and n_img == n_lbl else "  <-- check"
        print(f"  {split:5s}: {n_img} images / {n_lbl} labels{flag}")
        if n_img == 0 or n_img != n_lbl:
            ok = False
    return ok


def train(epochs=80, imgsz=640, batch=8, device=0,
          data_yaml=DATA_YAML, run_name="sign_detector_gtsdb", out_model=OUT_MODEL):
    """Fine-tune YOLOv8n on real sign data (GTSDB, or the merged German+Indian set)."""
    if not check_ready(data_yaml):
        print("\nDataset not ready — aborting before touching the GPU.")
        return None

    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"TRAINING: single-stage sign detector on {data_yaml.parent.name}")
    print(f"{'='*60}")
    print(f"  base=yolov8n.pt  epochs={epochs}  imgsz={imgsz}  batch={batch}")

    model = YOLO("yolov8n.pt")  # COCO-pretrained backbone
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,          # 640: GTSDB signs are small in 1360x800 frames;
                              # matches export_tensorrt.py's sign export path
        batch=batch,
        device=device,
        workers=2,
        patience=15,          # real data is small (~430 train), stop early if flat
        project=str(PROJECT_ROOT / "runs"),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        # Augmentation tuned for signs, not general COCO objects:
        mosaic=1.0,
        scale=0.5,            # signs appear across a wide scale range on the road
        fliplr=0.0,           # NEVER mirror — a mirrored arrow sign is a wrong sign
        flipud=0.0,
        hsv_v=0.4,            # lighting varies a lot; brightness jitter helps
        degrees=5.0,          # small rotation only; signs are near-upright
    )

    # Publish best weights under the name the export/pipeline expect
    best = PROJECT_ROOT / "runs" / run_name / "weights" / "best.pt"
    if not best.exists():
        best = best.with_name("last.pt")
    if best.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out_model)
        print(f"\n  Saved: {out_model}")
        print("  Next: python src/export_tensorrt.py   # build the .engine")
    else:
        print("\n  WARNING: no weights produced — training likely failed.")
    return out_model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train sign detector (GTSDB or merged)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=0, help="GPU id, or 'cpu' (slow, testing only)")
    p.add_argument("--data", default=None,
                   help="path to a dataset.yaml (default: German-only GTSDB). "
                        "Use data/signs_merged/dataset.yaml for German+Indian.")
    p.add_argument("--check", action="store_true",
                   help="verify dataset + print config, no training (no GPU needed)")
    a = p.parse_args()

    # If a custom --data is given, derive a distinct run name + output file so the
    # merged model does NOT overwrite german_sign_detector.pt.
    if a.data:
        data_yaml = Path(a.data)
        run_name = f"sign_detector_{data_yaml.parent.name}"
        out_model = MODELS_DIR / f"{data_yaml.parent.name}_detector.pt"
    else:
        data_yaml, run_name, out_model = DATA_YAML, "sign_detector_gtsdb", OUT_MODEL

    if a.check:
        print("Pre-flight check (no training):")
        ready = check_ready(data_yaml)
        print("\nREADY to train." if ready else "\nNOT ready — fix the above first.")
    else:
        train(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
              data_yaml=data_yaml, run_name=run_name, out_model=out_model)
