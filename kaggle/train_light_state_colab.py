"""Traffic-light STATE detector (red/green/yellow/off) on GOOGLE COLAB.

Colab twin of train_light_state_kaggle.py. Runs the SAME training on Colab's
free T4 so it can go in parallel with the Kaggle IDD run + the local sign run.

Differences from the Kaggle version (only the plumbing changes, not the model):
  - Colab has no "Add Data" mount, so we download the Bosch set via the Kaggle
    API using YOUR kaggle.json token (uploaded to the session).
  - Colab disk is ~78GB, and Bosch is small, so a normal download+unzip fits fine
    (no symlink trick needed like IDD's 19GB).
  - Model is copied to Google Drive at the end so it survives the session wipe.

HOW TO RUN (paste each block into its own Colab cell):

  # CELL 1 - GPU check + install
  !nvidia-smi -L
  !pip -q install ultralytics==8.4.120 kaggle

  # CELL 2 - upload your kaggle.json (from ~/.kaggle/kaggle.json on your PC)
  from google.colab import files
  files.upload()               # pick kaggle.json
  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

  # CELL 3 - (optional) mount Drive so the model survives disconnect
  from google.colab import drive
  drive.mount('/content/drive')

  # CELL 4 - paste THIS whole file and run

Then grab /content/light_state.pt (and the Drive copy if mounted).
"""

import os
import shutil
from pathlib import Path

# ------------------------------------------------------------------ config
IMGSZ = 640
BATCH = 16           # Colab free = ONE T4, so half the Kaggle 2x-T4 batch
EPOCHS = 60
SEED = 0
CLASS_NAMES = ["red", "green", "yellow", "off"]   # Bosch class_name.txt order
SLUG = "xingtingzhao/bosch-traffic-light-yolo"
RUN_NAME = "light_state"
OUT_NAME = "light_state.pt"

# Colab paths (not /kaggle/*). /content is the working disk (~78GB).
WORK = Path("/content")
DS = WORK / "light_state"
DL = WORK / "dl_bosch"
# If Drive is mounted, persist the model there too (survives session end).
DRIVE_OUT = Path("/content/drive/MyDrive/carlane_models")

# --- crash tolerance: resume from last.pt if a prior run left one ---
CKPT = WORK / "runs" / RUN_NAME / "weights" / "last.pt"
RESUMING = CKPT.exists()
if RESUMING:
    print(f"[resume] found {CKPT} -> continuing, skipping dataset rebuild")


# ------------------------------------------------- download dataset (Kaggle API)
def ensure():
    if DL.exists() and any(DL.rglob("*.png")):
        print("Bosch data already present at", DL)
        return
    print(f"Downloading {SLUG} via Kaggle API ...")
    DL.mkdir(parents=True, exist_ok=True)
    # requires ~/.kaggle/kaggle.json (uploaded in CELL 2)
    os.system(f"kaggle datasets download -d {SLUG} -p {DL} --unzip")

if not RESUMING:
    ensure()


# ------------------------------------------------------ locate images + labels
def find_split_dirs():
    img_root = lbl_root = None
    for p in DL.rglob("*"):
        if not p.is_dir():
            continue
        nm = p.name.lower()
        if img_root is None and nm == "images" and any(p.rglob("*.png")):
            img_root = p
        if lbl_root is None and nm == "labels" and any(p.rglob("*.txt")):
            lbl_root = p
    if img_root is None:
        for p in DL.rglob("*"):
            if p.is_dir() and any(p.glob("*.png")):
                img_root = p.parent if p.name.lower() in ("train", "val") else p
                break
    if lbl_root is None:
        for p in DL.rglob("*"):
            if p.is_dir() and any(p.glob("*.txt")) and "label" in str(p).lower():
                lbl_root = p.parent if p.name.lower() in ("train", "val") else p
                break
    return img_root, lbl_root


IMG_ROOT = LBL_ROOT = None
LBL_INDEX = {}
IMG_INDEX = {}
if not RESUMING:
    IMG_ROOT, LBL_ROOT = find_split_dirs()
    print("images root:", IMG_ROOT)
    print("labels root:", LBL_ROOT)
    if not (IMG_ROOT and LBL_ROOT):
        print("\n--- dirs with pngs/txts ---")
        for p in sorted(DL.rglob("*"))[:80]:
            if p.is_dir():
                n_png = len(list(p.glob("*.png")))
                n_txt = len(list(p.glob("*.txt")))
                if n_png or n_txt:
                    print(f"  {p}  ({n_png} png, {n_txt} txt)")
        raise SystemExit("Could not locate images + labels. Check the listing above.")

    LBL_INDEX = {p.stem: p for p in Path(LBL_ROOT).rglob("*.txt")}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in Path(IMG_ROOT).rglob(ext):
            IMG_INDEX.setdefault(p.stem, p)
    print(f"indexed {len(IMG_INDEX)} images, {len(LBL_INDEX)} labels")

    # GUARD: Bosch class ids must be within 0..3.
    _probe_ids = set()
    for i, (_, lp) in enumerate(LBL_INDEX.items()):
        for line in lp.read_text().splitlines():
            line = line.strip()
            if line:
                _probe_ids.add(int(float(line.split()[0])))
        if i >= 200:
            break
    print("probe class ids (first 200 labels):", sorted(_probe_ids))
    if _probe_ids and not _probe_ids.issubset({0, 1, 2, 3}):
        raise SystemExit(f"Label class ids {sorted(_probe_ids)} exceed Bosch's 0..3 - "
                         "wrong labels folder or a different class scheme.")


# ------------------------------------------------------ build normalised dataset
def infer_split(path):
    low = str(path).lower()
    if "val" in low or "valid" in low or "test" in low:
        return "val"
    return "train"


def build():
    for split in ("train", "val"):
        (DS / split / "images").mkdir(parents=True, exist_ok=True)
        (DS / split / "labels").mkdir(parents=True, exist_ok=True)
    counts = {i: 0 for i in range(4)}
    kept = {"train": 0, "val": 0}
    for stem, lp in LBL_INDEX.items():
        img = IMG_INDEX.get(stem)
        if img is None:
            continue
        lines = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            counts[int(float(ln.split()[0]))] = counts.get(int(float(ln.split()[0])), 0) + 1
        split = infer_split(lp)
        shutil.copy2(img, DS / split / "images" / img.name)
        (DS / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        kept[split] += 1

    if kept["val"] == 0 and kept["train"] > 0:
        import random
        rng = random.Random(SEED)
        for img in sorted((DS / "train" / "images").glob("*")):
            if rng.random() < 0.15:
                lbl = DS / "train" / "labels" / f"{img.stem}.txt"
                img.rename(DS / "val" / "images" / img.name)
                if lbl.exists():
                    lbl.rename(DS / "val" / "labels" / lbl.name)
                kept["val"] += 1
                kept["train"] -= 1

    (DS / "dataset.yaml").write_text(
        f"path: {DS}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    print(f"\n{'='*60}")
    print(f"LIGHT-STATE DATASET  train={kept['train']}  val={kept['val']}")
    print("boxes per class: "
          + ", ".join(f"{CLASS_NAMES[k]}={counts.get(k,0)}" for k in range(4)))
    print(f"{'='*60}")
    if kept["train"] == 0 or kept["val"] == 0:
        raise SystemExit("Empty split - dataset not found correctly.")


if not RESUMING:
    print("Building light-state dataset...")
    build()

# ------------------------------------------------------------------ train
import torch
from ultralytics import YOLO

n_gpu = torch.cuda.device_count()
device = 0 if n_gpu >= 1 else "cpu"
print(f"GPUs: {n_gpu} -> device={device}, batch={BATCH}")
if device == "cpu":
    print("!! No GPU detected. In Colab: Runtime -> Change runtime type -> T4 GPU.")

# Crash-proof: sync best.pt to Drive after EVERY epoch, not just at the end.
# A Colab cutoff mid-run then costs at most one epoch, never the whole run.
def _sync_best_to_drive(trainer):
    try:
        b = getattr(trainer, "best", None)
        if b and Path(b).exists() and Path("/content/drive/MyDrive").exists():
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b, DRIVE_OUT / OUT_NAME)
    except Exception as e:
        print(f"  (per-epoch Drive sync skipped: {e})")

if RESUMING:
    print(f"[resume] loading {CKPT}")
    model = YOLO(str(CKPT))
    model.add_callback("on_fit_epoch_end", _sync_best_to_drive)
    model.train(resume=True)
else:
    model = YOLO("yolov8n.pt")   # COCO-pretrained backbone knows road scenes
    model.add_callback("on_fit_epoch_end", _sync_best_to_drive)
    model.train(
        data=str(DS / "dataset.yaml"),
        epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        device=device, workers=2, patience=12,
        project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True,
        pretrained=True, cache=False,
        save_period=1,           # write last.pt every epoch -> crash-safe
        mosaic=1.0, scale=0.5,
        fliplr=0.5,              # h-flip fine - red light is red mirrored
        flipud=0.0, degrees=0.0,
        hsv_h=0.0,               # DO NOT jitter hue - colour IS the class here
        hsv_s=0.3, hsv_v=0.4,
    )

best = WORK / "runs" / RUN_NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, WORK / OUT_NAME)
    print(f"\nDONE. Download /content/{OUT_NAME}")
    # Persist to Drive if mounted (survives the Colab session wipe).
    try:
        if Path("/content/drive/MyDrive").exists():
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, DRIVE_OUT / OUT_NAME)
            print(f"  also saved to {DRIVE_OUT / OUT_NAME} (survives disconnect)")
    except Exception as e:
        print(f"  (Drive copy skipped: {e})")
else:
    print("\nWARNING: best.pt not found - check training logs.")
