"""Traffic-light STATE detector (Kaggle GPU) - red / green / yellow / off.

Why this exists: the pipeline's night eval flagged traffic lights as the weakest
stage (3.0 -> 1.0 detections/frame at night). Today lights come from YOLOP boxes
+ a colour heuristic. A dedicated detector that emits the colour STATE directly is
the honest fix - "we trained a light-state detector, here's its mAP" beats "we
threshold pixels".

Data: xingtingzhao/bosch-traffic-light-yolo - the Bosch Small Traffic Lights
Dataset in YOLO format, real dashcam frames, native classes (class_name.txt):
  0 red   1 green   2 yellow   3 off
Light heads are universal, so a detector trained here transfers to Indian footage
(a red lamp is a red lamp) - no India-specific light data needed.

Accelerator = GPU T4 (x2 if available). Internet ON (self-downloads the dataset +
base weights). Run All -> download /kaggle/working/light_state.pt from Output.

ponytail: keeps Bosch's own 4 classes as-is (red/green/yellow/off) - no remap,
the dataset already models exactly the states we want. Ceiling: 'off' includes
occluded/back-facing lights; if the pipeline only cares about R/Y/G it can ignore
class 3 at inference. Upgrade path = drop class 3 at train time if it hurts mAP.
"""

import os
import glob
import shutil
from pathlib import Path

# ------------------------------------------------------------------ config
IMGSZ = 640
BATCH = 32           # 2x T4; halved on a single GPU
EPOCHS = 60
SEED = 0
CLASS_NAMES = ["red", "green", "yellow", "off"]   # Bosch class_name.txt order

WORK = Path("/kaggle/working")
DS = WORK / "light_state"           # normalised YOLO dataset we point training at
INPUT = Path("/kaggle/input")
SLUG = "xingtingzhao/bosch-traffic-light-yolo"
RUN_NAME = "light_state"
OUT_NAME = "light_state.pt"

# --- crash tolerance: resume from last.pt if an interrupted run left one, and
# skip the dataset rebuild. last.pt is written every epoch (save_period=1),
# best.pt kept updated, so a crash never loses more than the in-progress epoch.
CKPT = WORK / "runs" / RUN_NAME / "weights" / "last.pt"
RESUMING = CKPT.exists()
if RESUMING:
    print(f"[resume] found {CKPT} -> continuing training, skipping dataset rebuild")


# ------------------------------------------------- ensure dataset available
EXTRA_ROOTS = []

def ensure():
    for p in INPUT.rglob("*"):
        if p.is_dir() and "bosch" in p.name.lower() and any(p.rglob("*.png")):
            print("Dataset appears attached under", p)
            return
    print("Dataset not attached - downloading via Kaggle API...")
    dl = WORK / "dl_bosch"
    dl.mkdir(exist_ok=True)
    os.system(f"kaggle datasets download -d {SLUG} -p {dl} --unzip")
    EXTRA_ROOTS.append(dl)

if not RESUMING:
    ensure()

def roots():
    return [INPUT] + EXTRA_ROOTS


# ------------------------------------------------------ locate images + labels
def find_split_dirs():
    """Find (images_root, labels_root). Bosch lays out images/{train,val} and
    labels/{train,val}; be robust to a flat layout too."""
    img_root = lbl_root = None
    for root in roots():
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            nm = p.name.lower()
            if img_root is None and nm == "images" and any(p.rglob("*.png") for _ in [0]):
                img_root = p
            if lbl_root is None and nm == "labels" and any(p.rglob("*.txt") for _ in [0]):
                lbl_root = p
    # fallback: any dir with pngs / any dir with txts
    if img_root is None:
        for root in roots():
            for p in root.rglob("*"):
                if p.is_dir() and any(p.glob("*.png")):
                    img_root = p.parent if p.name.lower() in ("train", "val") else p
                    break
            if img_root:
                break
    if lbl_root is None:
        for root in roots():
            for p in root.rglob("*"):
                if p.is_dir() and any(p.glob("*.txt")) and "label" in str(p).lower():
                    lbl_root = p.parent if p.name.lower() in ("train", "val") else p
                    break
            if lbl_root:
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
        print("\n--- dirs with pngs/txts across roots ---")
        for root in roots():
            for p in sorted(root.rglob("*"))[:80]:
                if p.is_dir():
                    n_png = len(list(p.glob("*.png")))
                    n_txt = len(list(p.glob("*.txt")))
                    if n_png or n_txt:
                        print(f"  {p}  ({n_png} png, {n_txt} txt)")
        raise SystemExit("Could not locate images + labels. Check the listing above.")

    # index every label by stem, and every image by stem
    LBL_INDEX = {p.stem: p for p in Path(LBL_ROOT).rglob("*.txt")}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in Path(IMG_ROOT).rglob(ext):
            IMG_INDEX.setdefault(p.stem, p)
    print(f"indexed {len(IMG_INDEX)} images, {len(LBL_INDEX)} labels")

    # GUARD: labels must reference class ids within 0..3 (Bosch's 4 states). If
    # we find ids outside that, we grabbed the wrong labels folder - fail loudly.
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
        # keep even empty-label frames? no - a light detector wants positives;
        # empty files are background, fine to include a few but skip to stay lean
        if not lines:
            continue
        for ln in lines:
            counts[int(float(ln.split()[0]))] = counts.get(int(float(ln.split()[0])), 0) + 1
        split = infer_split(lp)
        shutil.copy2(img, DS / split / "images" / img.name)
        (DS / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        kept[split] += 1

    # if the source had no val split, carve one out of train (deterministic)
    if kept["val"] == 0 and kept["train"] > 0:
        import random
        rng = random.Random(SEED)
        tr_imgs = sorted((DS / "train" / "images").glob("*"))
        for img in tr_imgs:
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
device = list(range(n_gpu)) if n_gpu > 1 else (0 if n_gpu == 1 else "cpu")
batch = BATCH if n_gpu >= 2 else max(8, BATCH // 2)
print(f"GPUs: {n_gpu} -> device={device}, batch={batch}")

if RESUMING:
    print(f"[resume] loading {CKPT}")
    model = YOLO(str(CKPT))
    model.train(resume=True)
else:
    model = YOLO("yolov8n.pt")   # COCO-pretrained backbone knows road scenes
    model.train(
        data=str(DS / "dataset.yaml"),
        epochs=EPOCHS, imgsz=IMGSZ, batch=batch,
        device=device, workers=4, patience=12,
        project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True,
        pretrained=True, cache=False,
        save_period=1,           # write last.pt every epoch -> crash-safe
        mosaic=1.0, scale=0.5,
        fliplr=0.5,              # horizontal flip is fine - red light is red mirrored
        flipud=0.0, degrees=0.0,
        hsv_h=0.0,               # DO NOT jitter hue - colour IS the class here
        hsv_s=0.3, hsv_v=0.4,    # sat/brightness jitter ok for day/night robustness
    )

best = WORK / "runs" / RUN_NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, WORK / OUT_NAME)
    print(f"\nDONE. Download /kaggle/working/{OUT_NAME} from the Output tab.")
    # Shrink the Kaggle output: drop built dataset + downloaded raw set.
    for scratch in (DS, WORK / "dl_bosch"):
        try:
            if scratch.exists():
                shutil.rmtree(scratch)
                print(f"  cleaned scratch: {scratch}")
        except Exception as e:
            print(f"  (could not remove {scratch}: {e})")
else:
    print("\nWARNING: best.pt not found - check training logs.")
