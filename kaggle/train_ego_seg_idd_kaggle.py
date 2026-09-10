"""Ego-lane drivable-area segmentation on INDIAN roads (IDD, Kaggle GPU).

Why: the local ego_seg trains on BDD (US roads). The showcase footage is now
Indian (Kolkata / Mumbai / NH-44). Training the same YOLOv8n-seg task on the
India Driving Dataset closes that domain gap so the drivable corridor is learned
on chaotic, often-unpainted Indian roads - the honest match for the demo.

Data: redzapdos123/indian-driving-dataset-segmentations-yolov11-seg
  Real IDD images (leftImg8bit), YOLO-seg POLYGON labels, train/val/test splits,
  a data.yaml with the class list. IDD-Seg has many classes (road, drivable
  fallback, sidewalk, ...). We collapse to ONE 'ego_lane' class = the drivable
  road surface, matching the local ego_seg (nc=1, names: 0 ego_lane) so the
  resulting .pt is a drop-in for the same pipeline runner.

Accelerator = GPU T4 (x2 if available). Internet ON (self-downloads dataset +
base weights). Run All -> download /kaggle/working/ego_seg_idd.pt from Output.

ponytail: the drivable class is auto-detected from data.yaml by name keyword
('drivable' or 'road'), printed for eyeballing, and every matched class id is
remapped to 0. Ceiling: if IDD's naming is unusual the keyword may pick the wrong
class - the printed mapping makes that visible; fix via DRIVABLE_OVERRIDE.
"""

import os
import shutil
from pathlib import Path

# ------------------------------------------------------------------ config
IMGSZ = 640
BATCH = 32
EPOCHS = 50
SEED = 0
RUN_NAME = "ego_seg_idd"
OUT_NAME = "ego_seg_idd.pt"
# IDD-Seg is ~16k images / 19GB. Kaggle /kaggle/working is only ~20GB, so we
# (a) cap how many images we build into the training set, and (b) SYMLINK images
# instead of copying them (a copy would double the disk use and overflow). A 12k
# subset is plenty for a single-class drivable model and keeps disk safe.
N_TRAIN = 12000
N_VAL = 2000
# Which IDD-Seg class name(s) count as the ego drivable surface. Keyword match,
# case-insensitive. 'drivable' is the primary; 'road' is the fallback if the
# dataset has no explicit drivable class.
DRIVABLE_KEYWORDS = ["drivable", "road"]
DRIVABLE_OVERRIDE = None   # set to an int class id to force it, if keywords miss

WORK = Path("/kaggle/working")
DS = WORK / "ego_seg_idd"
INPUT = Path("/kaggle/input")
SLUG = "redzapdos123/indian-driving-dataset-segmentations-yolov11-seg"

# --- crash tolerance ---------------------------------------------------------
# If a previous (interrupted) run left a checkpoint, we RESUME from it instead of
# rebuilding the dataset and restarting at epoch 1. Ultralytics writes last.pt
# after every epoch (save_period=1 below) and keeps best.pt updated whenever a
# new best appears, so a crash never loses more than the in-progress epoch and
# the best model so far is always on disk.
CKPT = WORK / "runs" / RUN_NAME / "weights" / "last.pt"
RESUMING = CKPT.exists()
if RESUMING:
    print(f"[resume] found {CKPT} -> continuing training, skipping dataset rebuild")


# ------------------------------------------------- ensure dataset available
EXTRA_ROOTS = []

def ensure():
    for p in INPUT.rglob("*"):
        if p.is_dir() and "idd" in p.name.lower() and any(p.rglob("*.png")):
            print("Dataset attached under", p, "(good - no download needed)")
            return
    # IDD-Seg is 19GB; downloading its zip AND unzipping it needs ~38GB, which
    # overflows Kaggle's ~20GB /kaggle/working (that was the No-space error).
    # The reliable fix is to ATTACH it (mounts read-only under /kaggle/input,
    # costs no working-disk). Fail loudly with instructions rather than retry a
    # download that cannot fit.
    raise SystemExit(
        "IDD-Seg dataset is not attached, and it is too large (19GB) to download+\n"
        "unzip inside /kaggle/working (~20GB). ATTACH it instead:\n"
        "  Right panel -> Add Data -> search\n"
        f"  '{SLUG.split('/')[-1]}' (by {SLUG.split('/')[0]}) -> Add.\n"
        "Then Run All again - the notebook auto-detects it under /kaggle/input.")

if not RESUMING:
    ensure()

def roots():
    return [INPUT] + EXTRA_ROOTS


# ------------------------------------------------------ locate dataset + classes
def find_yaml_and_dirs():
    for root in roots():
        if not root.exists():
            continue
        for y in root.rglob("data.yaml"):
            base = y.parent
            imgs = [q for q in base.rglob("*") if q.is_dir() and q.name == "images"]
            lbls = [q for q in base.rglob("*") if q.is_dir() and q.name == "labels"]
            if imgs and lbls:
                return y, base
    return None, None


YAML, BASE = (None, None) if RESUMING else find_yaml_and_dirs()
if not RESUMING:
  print("data.yaml:", YAML, "\nbase:", BASE)
if not RESUMING and not YAML:
    print("\n--- dirs with pngs across roots ---")
    for root in roots():
        for p in sorted(root.rglob("*"))[:80]:
            if p.is_dir() and any(p.glob("*.png")):
                print("  ", p, len(list(p.glob("*.png"))), "png")
    raise SystemExit("Could not find IDD-Seg data.yaml with images/ + labels/.")

# pick the drivable class id(s). Guard against NEGATED names: 'non-drivable
# fallback' contains the substring 'drivable' but is the opposite class - a
# naive substring match would train on the non-drivable area too.
NEG_MARKERS = ["non-drivable", "non drivable", "nondrivable", "not drivable"]

def _matches(name, kw):
    low = str(name).lower()
    if any(neg in low for neg in NEG_MARKERS):
        return False
    return kw in low

def detect_drivable_ids():
    import yaml as _yaml
    cfg = _yaml.safe_load(Path(YAML).read_text())
    raw = cfg.get("names")
    if isinstance(raw, dict):
        names = [raw[k] for k in sorted(raw, key=lambda x: int(x))]
    else:
        names = list(raw)
    print(f"IDD-Seg classes ({len(names)}):")
    for i, n in enumerate(names):
        print(f"  {i:2d} {n}")
    if DRIVABLE_OVERRIDE is not None:
        ids = {int(DRIVABLE_OVERRIDE)}
    else:
        ids = set()
        for kw in DRIVABLE_KEYWORDS:
            for i, n in enumerate(names):
                if _matches(n, kw):
                    ids.add(i)
            if ids:   # stop at the first keyword that matched anything
                break
    print("\n--- drivable class id(s) chosen (remap -> 0 ego_lane):",
          sorted(ids), "=", [names[i] for i in sorted(ids)])
    if not ids:
        raise SystemExit("No drivable/road class found by keyword. Inspect the "
                         "class list above and set DRIVABLE_OVERRIDE.")
    return ids


# ------------------------------------------------------ build single-class set
def split_of(path):
    low = str(path).lower()
    if "val" in low or "valid" in low:
        return "val"
    if "test" in low:
        return "val"   # fold test into val for a bigger eval set
    return "train"


def build(drivable_ids):
    for split in ("train", "val"):
        (DS / split / "images").mkdir(parents=True, exist_ok=True)
        (DS / split / "labels").mkdir(parents=True, exist_ok=True)
    img_index = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in Path(BASE).rglob(ext):
            if "/images" in str(p).replace("\\", "/").lower() or p.parent.name == "images" \
               or "images" in [x.lower() for x in p.parts]:
                img_index.setdefault(p.stem, p)
    labels = [lp for lp in Path(BASE).rglob("*.txt")
              if lp.name.lower() != "classes.txt" and lp.stem in img_index]
    # deterministic shuffle so the N_TRAIN/N_VAL caps sample across the whole set
    import random as _r
    _r.Random(SEED).shuffle(labels)
    caps = {"train": N_TRAIN, "val": N_VAL}
    kept = {"train": 0, "val": 0}
    polys = {"train": 0, "val": 0}

    if not img_index:
        raise SystemExit("No images found under the dataset base - is it attached "
                         "correctly under /kaggle/input? See the class listing above.")

    # Verify symlinks actually work here BEFORE building 12k of them. IDD is
    # 19GB; if symlinks silently failed we'd copy 19GB and overflow /kaggle/
    # working. Probe once: create+read a link, and confirm it resolves.
    _probe = DS / "_symlink_probe"
    _target = next(iter(img_index.values()))
    SYMLINKS_OK = False
    try:
        if _probe.exists() or _probe.is_symlink():
            _probe.unlink()
        os.symlink(_target, _probe)
        SYMLINKS_OK = _probe.resolve().exists()
        _probe.unlink()
    except OSError:
        SYMLINKS_OK = False
    print(f"symlink support: {'OK (zero-copy build)' if SYMLINKS_OK else 'NO'}")
    if not SYMLINKS_OK:
        raise SystemExit(
            "Symlinks unavailable on this filesystem. Copying 19GB of IDD images\n"
            "into /kaggle/working would overflow its ~20GB quota. Options:\n"
            "  - lower N_TRAIN/N_VAL at the top so the COPIED subset fits, then\n"
            "    change _place() to shutil.copy2, OR\n"
            "  - run where symlinks are allowed.")

    def _place(src, dst):
        """Symlink the image (zero extra disk). Symlink support was verified
        above, so a failure here is a real error worth surfacing."""
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    for lp in labels:
        img = img_index.get(lp.stem)
        if img is None:
            continue
        sp = split_of(lp)
        if kept[sp] >= caps[sp]:
            continue   # respect the per-split cap to stay within disk quota
        out_lines = []
        for line in lp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if cid in drivable_ids and len(parts) >= 7:   # class + >=3 xy pairs
                out_lines.append("0 " + " ".join(parts[1:]))
        _place(img, DS / sp / "images" / img.name)
        (DS / sp / "labels" / f"{lp.stem}.txt").write_text("\n".join(out_lines))
        kept[sp] += 1
        if out_lines:
            polys[sp] += 1
        if kept["train"] >= caps["train"] and kept["val"] >= caps["val"]:
            break
    (DS / "dataset.yaml").write_text(
        f"path: {DS}\ntrain: train/images\nval: val/images\n\n"
        f"nc: 1\nnames:\n  0: ego_lane\n")
    print(f"\n{'='*60}")
    print(f"IDD EGO-LANE DATASET  train={kept['train']} ({polys['train']} with polygon)  "
          f"val={kept['val']} ({polys['val']} with polygon)")
    print(f"{'='*60}")
    if kept["train"] == 0 or kept["val"] == 0:
        raise SystemExit("Empty split - dataset not found correctly.")
    if polys["train"] == 0:
        raise SystemExit("No drivable polygons kept - wrong class id? See mapping above.")


if not RESUMING:
    drivable_ids = detect_drivable_ids()
    print("Building IDD ego-lane (single-class) dataset...")
    build(drivable_ids)

# ------------------------------------------------------------------ train
import torch
from ultralytics import YOLO

n_gpu = torch.cuda.device_count()
device = list(range(n_gpu)) if n_gpu > 1 else (0 if n_gpu == 1 else "cpu")
batch = BATCH if n_gpu >= 2 else max(16, BATCH // 2)
print(f"GPUs: {n_gpu} -> device={device}, batch={batch}")

if RESUMING:
    # continue the interrupted run from its last checkpoint; all training args
    # (data, epochs, aug, ...) are restored from the checkpoint itself.
    print(f"[resume] loading {CKPT}")
    model = YOLO(str(CKPT))
    model.train(resume=True)
else:
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(DS / "dataset.yaml"),
        epochs=EPOCHS, imgsz=IMGSZ, batch=batch,
        device=device, workers=4, patience=12,
        project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True,
        pretrained=True, cache=False,
        save_period=1,   # write last.pt every epoch -> crash never loses >1 epoch
        fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.5, hsv_v=0.4, mosaic=1.0,
    )

best = WORK / "runs" / RUN_NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, WORK / OUT_NAME)
    print(f"\nDONE. Download /kaggle/working/{OUT_NAME} from the Output tab.")
    # Shrink the Kaggle output: drop built dataset + downloaded raw IDD set.
    for scratch in (DS, WORK / "dl_idd"):
        try:
            if scratch.exists():
                shutil.rmtree(scratch)
                print(f"  cleaned scratch: {scratch}")
        except Exception as e:
            print(f"  (could not remove {scratch}: {e})")
else:
    print("\nWARNING: best.pt not found - check training logs.")
