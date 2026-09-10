"""
CarLaneI — Ego-Lane Segmentation training on Kaggle (2x T4).

HOW TO USE (browser):
  1. New Notebook on kaggle.com -> Settings -> Accelerator = "GPU T4 x2".
  2. Add Data -> search & attach the BDD100K drivable dataset that contains
     drivable-area MASKS (the same one used locally:
     "nguyentrongquocdat/object-detection-and-drivable-lane-masking").
  3. Paste this whole file into a cell (or upload as the notebook) and Run All.
  4. When done, download /kaggle/working/ego_seg.pt  (Output tab) and drop it
     into your local  models/  folder. Re-export the TensorRT engine locally.

It AUTO-DETECTS the images + drivable masks anywhere under /kaggle/input, builds a
YOLOv8-seg dataset for the EGO lane (BDD class 0 = direct drivable), and trains
with a larger config than the laptop run:
  ~25-30k images, imgsz 640, batch 32 (2x T4 = 32GB), ~40 epochs -> ~one session.
"""

import os, glob, random, shutil
from pathlib import Path
import cv2, numpy as np

# ------------------------------------------------------------------ config
N_TRAIN   = 28000      # images to sample for training (fits one session)
N_VAL     = 2000
IMGSZ     = 640
BATCH     = 32         # 2x T4 (32GB) handles this for yolov8n-seg
EPOCHS    = 40
EGO_VALUE = 0          # BDD drivable mask: 0=direct(ego), 1=alt, 2=background
MIN_AREA_FRAC = 0.004
MAX_POLYS = 3
POLY_EPS_FRAC = 0.004

WORK = Path("/kaggle/working")
DS   = WORK / "ego_seg"
INPUT = Path("/kaggle/input")
RUN_NAME = "ego_seg"
OUT_NAME = "ego_seg.pt"

# --- crash tolerance ---------------------------------------------------------
# If a previous (interrupted) run left a checkpoint, RESUME from it instead of
# rebuilding the dataset and restarting at epoch 1. Ultralytics writes last.pt
# every epoch (save_period=1) and keeps best.pt updated, so a crash never loses
# more than the in-progress epoch and the best model so far is always on disk.
CKPT = WORK / "runs" / RUN_NAME / "weights" / "last.pt"
RESUMING = CKPT.exists()
if RESUMING:
    print(f"[resume] found {CKPT} -> continuing training, skipping dataset rebuild")

# --------------------------------------------- ensure the dataset is available
# Works whether the dataset was ATTACHED via the UI (under /kaggle/input) OR not
# (falls back to downloading it with the Kaggle API using the kernel's token).
BDD_SLUG = "nguyentrongquocdat/object-detection-and-drivable-lane-masking"

def ensure_dataset():
    # already attached?
    for p in INPUT.rglob("*"):
        if p.is_dir() and "drivable" in " ".join(x.lower() for x in p.parts) and any(p.glob("*.png")):
            print("Dataset already attached under /kaggle/input.")
            return
    print("Dataset not attached — downloading via Kaggle API...")
    try:
        import kaggle  # kernel has creds if Internet is ON
        dl = WORK / "bdd_dl"; dl.mkdir(exist_ok=True)
        os.system(f"kaggle datasets download -d {BDD_SLUG} -p {dl} --unzip")
        # point INPUT-style search at the download dir too
        globals()["EXTRA_ROOTS"] = [dl]
    except Exception as e:
        print("Auto-download failed:", e,
              "\n-> Attach the dataset via Add Data instead.")

EXTRA_ROOTS = []
if not RESUMING:
    ensure_dataset()

# ------------------------------------------------------ locate data on Kaggle
def _roots():
    return [INPUT] + [Path(r) for r in EXTRA_ROOTS]

def find_dirs():
    """Find the BDD images dir and the drivable MASK dir under any data root."""
    mask_dir = img_dir = None
    mask_candidates = []
    for root in _roots():
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            name = p.name.lower()
            parts = " ".join(x.lower() for x in p.parts)
            # collect any PNG folder on a drivable path, EXCLUDING colormaps/polygons
            if ("drivable" in parts or "mask" in name) and any(p.glob("*.png")):
                if "colormap" not in parts and "polygon" not in parts:
                    mask_candidates.append(p)
            if img_dir is None and any(p.glob("*.jpg")) and "train" in name:
                img_dir = p
    # prefer a folder literally called 'masks', then a 'train' split
    def _rank(p):
        s = " ".join(x.lower() for x in p.parts)
        return (("mask" in p.name.lower()) * 2 + ("train" in s) * 1)
    if mask_candidates:
        mask_dir = sorted(mask_candidates, key=_rank, reverse=True)[0]
    if img_dir is None:
        for root in _roots():
            for p in root.rglob("*"):
                if p.is_dir() and any(p.glob("*.jpg")):
                    img_dir = p; break
            if img_dir:
                break
    return img_dir, mask_dir

IMG_DIR = MASK_DIR = None
IMG_INDEX = {}
if not RESUMING:
    IMG_DIR, MASK_DIR = find_dirs()
    print("images dir:", IMG_DIR)
    print("masks  dir:", MASK_DIR)
    if not (IMG_DIR and MASK_DIR):
        print("\n--- folders with images/masks across data roots ---")
        for root in _roots():
            for p in sorted(root.rglob("*"))[:60]:
                if p.is_dir():
                    n_png = len(list(p.glob('*.png'))); n_jpg = len(list(p.glob('*.jpg')))
                    if n_png or n_jpg:
                        print(f"  {p}  ({n_jpg} jpg, {n_png} png)")
        raise SystemExit(
            "Could not find images + drivable MASKS. Attach the dataset that has "
            "labels/drivable_labels/masks/*.png "
            "(nguyentrongquocdat/object-detection-and-drivable-lane-masking).")

    # GUARD: the mask must be a LABEL map (values in {0,1,2}), not an RGB colormap.
    # colormaps/ look like masks (PNGs on a 'drivable' path) but 'mask==0' would
    # be meaningless there. Refuse to train on the wrong folder.
    _probe = cv2.imread(str(next(Path(MASK_DIR).glob("*.png"))), cv2.IMREAD_UNCHANGED)
    _vals = set(np.unique(_probe).tolist())
    print("mask sample unique values:", sorted(_vals)[:8], "shape:", _probe.shape)
    if not _vals.issubset({0, 1, 2}):
        raise SystemExit(
            f"MASK_DIR {MASK_DIR} does not look like a label map (values {sorted(_vals)[:8]}). "
            "It's probably a colormap/visualisation. Point to labels/drivable_labels/masks/.")

    # filename->image path index (images may be split across train/val dirs)
    for jp in Path(IMG_DIR.parent).rglob("*.jpg"):
        IMG_INDEX.setdefault(jp.stem, jp)
    print("indexed images:", len(IMG_INDEX))

# ------------------------------------------------------ mask -> YOLO polygons
def mask_to_polys(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = mask.shape
    ego = (mask == EGO_VALUE).astype(np.uint8)
    if ego.sum() < MIN_AREA_FRAC * h * w:
        return []
    ego = cv2.morphologyEx(ego, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(ego, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:MAX_POLYS]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA_FRAC * h * w:
            continue
        eps = POLY_EPS_FRAC * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(ap) < 3:
            continue
        ap[:, 0] /= w; ap[:, 1] /= h
        out.append(np.clip(ap, 0, 1).reshape(-1).tolist())
    return out

def build():
    masks = sorted(Path(MASK_DIR).glob("*.png"))
    random.Random(0).shuffle(masks)
    for split, n in (("train", N_TRAIN), ("val", N_VAL)):
        (DS / split / "images").mkdir(parents=True, exist_ok=True)
        (DS / split / "labels").mkdir(parents=True, exist_ok=True)
    train_masks = masks[:N_TRAIN]
    val_masks   = masks[N_TRAIN:N_TRAIN + N_VAL]
    for split, subset in (("train", train_masks), ("val", val_masks)):
        kept = 0
        for i, mp in enumerate(subset):
            img = IMG_INDEX.get(mp.stem)
            if img is None:
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            polys = mask_to_polys(m)
            shutil.copy2(img, DS / split / "images" / img.name)
            lbl = DS / split / "labels" / f"{mp.stem}.txt"
            if polys:
                lbl.write_text("\n".join("0 " + " ".join(f"{v:.6f}" for v in p) for p in polys))
                kept += 1
            else:
                lbl.write_text("")
            if (i + 1) % 5000 == 0:
                print(f"  {split}: {i+1}/{len(subset)}")
        print(f"{split}: {kept} with ego polygon (of {len(subset)})")
    (DS / "dataset.yaml").write_text(
        f"path: {DS}\ntrain: train/images\nval: val/images\n\nnc: 1\nnames:\n  0: ego_lane\n")

if not RESUMING:
    print("Building dataset (this takes a few minutes)...")
    build()

# ------------------------------------------------------------------ train
import torch
from ultralytics import YOLO

n_gpu = torch.cuda.device_count()
device = list(range(n_gpu)) if n_gpu > 1 else (0 if n_gpu == 1 else "cpu")
# scale batch to GPU count (2x T4 = 32GB can take more)
batch = BATCH if n_gpu >= 2 else max(16, BATCH // 2)
print(f"GPUs: {n_gpu} -> device={device}, batch={batch}")

if RESUMING:
    print(f"[resume] loading {CKPT}")
    model = YOLO(str(CKPT))
    model.train(resume=True)
else:
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(DS / "dataset.yaml"),
        epochs=EPOCHS, imgsz=IMGSZ, batch=batch,
        device=device,
        workers=4, patience=12, project=str(WORK / "runs"), name=RUN_NAME,
        exist_ok=True, pretrained=True, cache=False,
        save_period=1,   # write last.pt every epoch -> crash never loses >1 epoch
        fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.5, hsv_v=0.4, mosaic=1.0,
    )

best = WORK / "runs" / RUN_NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, WORK / OUT_NAME)
    print(f"\nDONE. Download /kaggle/working/{OUT_NAME} from the Output tab.")

    # Shrink the Kaggle OUTPUT: the built dataset + downloaded raw data are ~8GB
    # of scratch we never need back. Delete them so the committed output is just
    # the model (~7MB) + runs/ (curves, results.csv). Only runs after a good save.
    for scratch in (DS, WORK / "bdd_dl"):
        try:
            if scratch.exists():
                shutil.rmtree(scratch)
                print(f"  cleaned scratch: {scratch}")
        except Exception as e:
            print(f"  (could not remove {scratch}: {e})")
else:
    print("\nWARNING: best.pt not found — check training logs.")
