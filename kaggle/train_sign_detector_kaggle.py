"""Merged Indian + German traffic-sign DETECTOR (Kaggle GPU).

One detector, two real datasets, mapped into the SAME 4 shape+colour super-classes
the local pipeline already uses (prohibitory / mandatory / danger / other):

  German : sovitrath/gtsdb-dataset-in-pascal-voc-structure   (real GTSDB road photos, VOC)
  Indian : akbaralibatti/indian-traffic-sign-yolo11          (7.5k real dashcam photos, YOLO)

Why super-classes and not specific sign names: a red circle is a prohibition in
Delhi and Munich alike; naming *which* prohibition is a guess the shape+colour
can't back (that was the old saturated-classifier failure). 4 super-classes are
what the data honestly supports across both countries, and it makes the two
datasets directly mergeable despite different label schemes.

Accelerator = GPU T4 (x2 if available). Internet ON (self-downloads both datasets
+ base weights). Run All -> download /kaggle/working/sign_detector.pt from Output.

ponytail: Indian class->superclass is by keyword on the dataset's own class names,
printed for eyeballing; unmatched names fall to 'other' (honest default, never a
specific guess). Ceiling: a mislabelled/oddly-named Indian class lands in 'other'
instead of its true bucket. Upgrade path = hand-map that name in INDIAN_OVERRIDES.
"""

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ------------------------------------------------------------------ config
IMGSZ = 640          # GTSDB signs are small in 1360x800; matches local export path
BATCH = 32           # 2x T4 (32GB); halved automatically on a single GPU
EPOCHS = 80
VAL_FRAC = 0.15      # for the German split (Indian set ships its own train/val)
SEED = 0

WORK = Path("/kaggle/working")
DS = WORK / "signs_merged"          # unified YOLO dataset we build
INPUT = Path("/kaggle/input")
RUN_NAME = "sign_detector"
OUT_NAME = "sign_detector.pt"

GTSDB_SLUG = "sovitrath/gtsdb-dataset-in-pascal-voc-structure"
INDIAN_SLUG = "akbaralibatti/indian-traffic-sign-yolo11"

# --- crash tolerance: resume from last.pt if an interrupted run left one, and
# skip the dataset rebuild. last.pt is written every epoch (save_period=1),
# best.pt kept updated, so a crash never loses more than the in-progress epoch.
CKPT = WORK / "runs" / RUN_NAME / "weights" / "last.pt"
RESUMING = CKPT.exists()
if RESUMING:
    print(f"[resume] found {CKPT} -> continuing training, skipping dataset rebuild")

SUPERCLASS_NAMES = ["prohibitory", "mandatory", "danger", "other"]

# --- German GTSDB: 43 fine classes -> 4 super (identical to prepare_gtsdb_yolo.py)
FINE_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]
FINE_INDEX = {n: i for i, n in enumerate(FINE_NAMES)}


def gtsdb_super(fine_id):
    PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
    MANDATORY = {33, 34, 35, 36, 37, 38, 39, 40}
    DANGER = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    if fine_id in PROHIBITORY:
        return 0
    if fine_id in MANDATORY:
        return 1
    if fine_id in DANGER:
        return 2
    return 3


# --- Indian: map a raw class NAME to a super-class by shape/colour keywords.
# Rules ordered; first match wins. Names come from the dataset's own data.yaml.
INDIAN_OVERRIDES = {}  # exact-name -> super id, for anything the keywords miss

def indian_super(name):
    n = name.lower().strip()
    if n in INDIAN_OVERRIDES:
        return INDIAN_OVERRIDES[n]
    # DANGER: red triangle warnings
    danger_kw = ["warning", "caution", "curve", "bend", "slippery", "narrow",
                 "hump", "bump", "dip", "ripple", "gap", "cattle", "pedestrian",
                 "school", "children", "cross", "round about", "roundabout ahead",
                 "junction", "gap in median", "falling", "cycle cross", "animal",
                 "men at work", "road work", "barrier", "ferry", "ford",
                 "steep", "descent", "ascent", "hairpin", "loose", "guard"]
    # MANDATORY: blue circle / compulsory
    mandatory_kw = ["compulsory", "mandatory", "ahead only", "turn right",
                    "turn left", "keep left", "keep right", "go straight",
                    "compulsory ahead", "sound horn", "cycle track",
                    "pedestrian only", "bus", "direction"]
    # PROHIBITORY: red circle / no- / restriction / speed limit
    prohib_kw = ["no ", "prohibited", "restriction", "speed limit", "no entry",
                 "overtaking prohibited", "horn prohibited", "no parking",
                 "no stopping", "u turn prohibited", "no left", "no right",
                 "axle load", "width limit", "height limit", "length limit",
                 "load limit", "max speed", "speed"]
    for kw in danger_kw:
        if kw in n:
            return 2
    for kw in mandatory_kw:
        if kw in n:
            return 1
    for kw in prohib_kw:
        if kw in n:
            return 0
    return 3  # other: priority/stop/yield/informatory/unmatched


# ------------------------------------------------- ensure datasets available
EXTRA_ROOTS = []

def ensure(slug, tag):
    for p in INPUT.rglob("*"):
        if p.is_dir() and slug.split("/")[-1].split("-")[0] in p.name.lower():
            print(f"[{tag}] appears attached under {p}")
            return
    print(f"[{tag}] downloading {slug} ...")
    dl = WORK / ("dl_" + tag)
    dl.mkdir(exist_ok=True)
    os.system(f"kaggle datasets download -d {slug} -p {dl} --unzip")
    EXTRA_ROOTS.append(dl)

if not RESUMING:
    ensure(GTSDB_SLUG, "gtsdb")
    ensure(INDIAN_SLUG, "indian")


def roots():
    return [INPUT] + EXTRA_ROOTS


# ------------------------------------------------------ locate the two datasets
def find_gtsdb_voc():
    """Return (Annotations dir, JPEGImages dir) for the German VOC dump."""
    for root in roots():
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and p.name == "Annotations" and any(p.glob("*.xml")):
                jpg = p.parent / "JPEGImages"
                if not jpg.exists():
                    # some dumps keep images beside annotations or in 'images'
                    cand = [q for q in p.parent.rglob("*")
                            if q.is_dir() and (any(q.glob("*.jpg")) or any(q.glob("*.ppm")))]
                    jpg = cand[0] if cand else p.parent
                return p, jpg
    return None, None


def find_indian_yolo():
    """Return (images_root, labels_root, data_yaml_path) for the Indian YOLO set."""
    yaml_path = None
    for root in roots():
        if not root.exists():
            continue
        for p in root.rglob("data.yaml"):
            # prefer one whose sibling has images/ + labels/
            base = p.parent
            if (base / "images").exists() or any(base.rglob("labels")):
                yaml_path = p
                break
        if yaml_path:
            break
    if not yaml_path:
        return None, None, None
    base = yaml_path.parent
    img_dirs = [q for q in base.rglob("*") if q.is_dir() and q.name == "images"]
    lbl_dirs = [q for q in base.rglob("*") if q.is_dir() and q.name == "labels"]
    return (img_dirs[0].parent if img_dirs else base), base, yaml_path


# --------------------------------------------------------- build unified dataset
def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    cl = lambda v: max(0.0, min(1.0, v))
    return cl(cx), cl(cy), cl(bw), cl(bh)


def add_german(counts):
    ann, jpg = find_gtsdb_voc()
    print("GTSDB Annotations:", ann, "\nGTSDB Images:", jpg)
    if not ann:
        print("  !! GTSDB VOC not found; skipping German data")
        return
    rng = random.Random(SEED)
    xmls = sorted(ann.glob("*.xml"))
    for xml in xmls:
        tree = ET.parse(xml)
        r = tree.getroot()
        size = r.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        stem = Path(r.find("filename").text).stem
        lines = []
        for obj in r.findall("object"):
            nm = obj.find("name").text.strip()
            fid = FINE_INDEX.get(nm)
            # GTSDB VOC dumps sometimes store the numeric class id as the name
            if fid is None and nm.isdigit() and int(nm) < 43:
                fid = int(nm)
            if fid is None:
                continue
            cls = gtsdb_super(fid)
            b = obj.find("bndbox")
            cx, cy, bw, bh = voc_to_yolo(
                float(b.find("xmin").text), float(b.find("ymin").text),
                float(b.find("xmax").text), float(b.find("ymax").text), w, h)
            if bw > 0 and bh > 0:
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts[cls] = counts.get(cls, 0) + 1
        if not lines:
            continue
        src = jpg / f"{stem}.jpg"
        if not src.exists():
            alt = jpg / f"{stem}.ppm"
            if alt.exists():
                src = alt
            else:
                hits = list(jpg.rglob(f"{stem}.*"))
                if not hits:
                    continue
                src = hits[0]
        split = "val" if rng.random() < VAL_FRAC else "train"
        # normalise .ppm -> .jpg so ultralytics reads it
        import cv2
        out_img = DS / split / "images" / f"de_{stem}.jpg"
        if src.suffix.lower() == ".ppm":
            im = cv2.imread(str(src))
            cv2.imwrite(str(out_img), im)
        else:
            shutil.copy2(src, out_img)
        (DS / split / "labels" / f"de_{stem}.txt").write_text("\n".join(lines))


def add_indian(counts):
    import yaml as _yaml
    img_root, lbl_root, yaml_path = find_indian_yolo()
    print("\nIndian images root:", img_root, "\nIndian data.yaml:", yaml_path)
    if not yaml_path:
        print("  !! Indian YOLO set not found; skipping Indian data")
        return
    cfg = _yaml.safe_load(Path(yaml_path).read_text())
    raw = cfg.get("names")
    if isinstance(raw, dict):
        names = [raw[k] for k in sorted(raw, key=lambda x: int(x))]
    else:
        names = list(raw)
    remap = {i: indian_super(nm) for i, nm in enumerate(names)}
    print("\n--- Indian class -> superclass mapping (eyeball this) ---")
    for i, nm in enumerate(names):
        print(f"  {i:3d} {nm:40s} -> {SUPERCLASS_NAMES[remap[i]]}")

    # pair every label .txt with its image, keep the dataset's own split if present
    all_labels = [p for p in Path(lbl_root).rglob("*.txt")]
    print(f"\nIndian label files: {len(all_labels)}")
    rng = random.Random(SEED)
    for lp in all_labels:
        # find the image with the same stem
        stem = lp.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = list(Path(img_root).rglob(f"{stem}{ext}"))
            if cand:
                img = cand[0]
                break
        if img is None:
            continue
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
            if cid not in remap:
                continue
            sup = remap[cid]
            out_lines.append(" ".join([str(sup)] + parts[1:]))
            counts[sup] = counts.get(sup, 0) + 1
        if not out_lines:
            continue
        # infer split from path, else random
        sp = "train"
        low = str(lp).lower()
        if "val" in low or "valid" in low:
            sp = "val"
        elif "test" in low:
            sp = "val"
        elif rng.random() < VAL_FRAC:
            sp = "val"
        shutil.copy2(img, DS / sp / "images" / f"in_{stem}{img.suffix}")
        (DS / sp / "labels" / f"in_{stem}.txt").write_text("\n".join(out_lines))


def build():
    for split in ("train", "val"):
        (DS / split / "images").mkdir(parents=True, exist_ok=True)
        (DS / split / "labels").mkdir(parents=True, exist_ok=True)
    counts = {}
    add_german(counts)
    add_indian(counts)
    (DS / "dataset.yaml").write_text(
        f"path: {DS}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(SUPERCLASS_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(SUPERCLASS_NAMES)))
    n_tr = len(list((DS / "train" / "images").glob("*")))
    n_va = len(list((DS / "val" / "images").glob("*")))
    print(f"\n{'='*60}")
    print(f"MERGED SIGN DATASET  train={n_tr} images  val={n_va} images")
    print(f"boxes per superclass: "
          + ", ".join(f"{SUPERCLASS_NAMES[k]}={counts.get(k,0)}" for k in range(4)))
    print(f"{'='*60}")
    if n_tr == 0 or n_va == 0:
        raise SystemExit("Empty split - neither dataset was found. Check the logs above.")


if not RESUMING:
    print("Building merged Indian+German sign dataset...")
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
        device=device, workers=4, patience=15,
        project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True,
        pretrained=True, cache=False,
        save_period=1,            # write last.pt every epoch -> crash-safe
        mosaic=1.0, scale=0.5,
        fliplr=0.0, flipud=0.0,   # NEVER mirror: a mirrored arrow sign is a wrong sign
        hsv_v=0.4, degrees=5.0,   # lighting varies; signs are near-upright
    )

best = WORK / "runs" / RUN_NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, WORK / "sign_detector.pt")
    print("\nDONE. Download /kaggle/working/sign_detector.pt from the Output tab.")
    # Shrink the Kaggle output: drop the built dataset + downloaded raw sets so
    # the commit is just the model + runs/, not multi-GB scratch.
    for scratch in (DS, WORK / "dl_gtsdb", WORK / "dl_indian"):
        try:
            if scratch.exists():
                shutil.rmtree(scratch)
                print(f"  cleaned scratch: {scratch}")
        except Exception as e:
            print(f"  (could not remove {scratch}: {e})")
else:
    print("\nWARNING: best.pt not found - check training logs.")
