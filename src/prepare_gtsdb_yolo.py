"""Convert the real GTSDB (Pascal VOC) dataset into YOLO detection format.

This replaces the synthetic paste-up dataset (train_combined_detector.py), which
was the root cause of the sign detector's ~57% real-world presence: it learned
GTSRB crops alpha-blended onto BDDA frames, not signs as they actually appear on
a real road (motion blur, real lighting, real backgrounds, real scale). GTSDB is
506 genuine 1360x800 road photographs with hand-drawn boxes — the honest data.

Class granularity — why 4 super-classes by default:
  GTSDB labels 43 fine classes but only ~1-2 boxes per image, so most classes get
  a handful of examples. A detector trained on that reproduces the exact failure
  the README documents: confident, specific, and wrong. The 4 GTSDB super-classes
  (prohibitory / mandatory / danger / other) are what sign SHAPE + COLOUR actually
  support reliably, and are standard for GTSDB detection. A red circle really is a
  prohibition; naming which prohibition is the classifier's saturated guess.

  --classes 43  keeps the fine labels if you have the data to back them up.

VOC box (xmin,ymin,xmax,ymax, absolute px) -> YOLO (cls cx cy w h, normalised).
Split is per-image 85/15, seeded, so train/val never share an image.

ponytail: 4 super-classes trade specific sign names for detections you can trust.
Ceiling — it won't tell "Stop" from "Yield", both are 'other'. Upgrade path = add
real per-class data (Mapillary MTSD) and switch to --classes 43.

Self-check:  python src/prepare_gtsdb_yolo.py --selftest
Convert:     python src/prepare_gtsdb_yolo.py
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT

VOC_ROOT = PROJECT_ROOT / "data" / "gtsdb_voc" / "input" / "data_root" / "dataset"
OUT_ROOT = PROJECT_ROOT / "data" / "gtsdb_yolo"

SUPERCLASS_NAMES = ["prohibitory", "mandatory", "danger", "other"]

# The 43 GTSDB/GTSRB classes in canonical order (index 0..42), matching the
# order in classes_list.txt after dropping '__background__'.
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
FINE_INDEX = {name: i for i, name in enumerate(FINE_NAMES)}


def superclass_of(fine_id):
    """Map a fine GTSDB class id (0..42) to a 4-way super-class id.

    This is the standard GTSDB detection grouping by sign shape+colour, not the
    grouping the old synthetic script used (that one mislabelled every triangle
    as danger AND swallowed stop/yield/priority into danger, leaving 'other'
    empty — the selftest catches exactly that). Explicit sets, no fragile ranges.
    """
    PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}   # red circle
    MANDATORY = {33, 34, 35, 36, 37, 38, 39, 40}            # blue circle
    DANGER = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,   # red triangle
              28, 29, 30, 31}
    if fine_id in PROHIBITORY:
        return 0
    if fine_id in MANDATORY:
        return 1
    if fine_id in DANGER:
        return 2
    return 3  # other: end-limits(6,32,41,42), priority(12), yield(13),
              #        stop(14), no entry(17)


def _voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    """VOC absolute corners -> YOLO normalised centre/size. Clamped to [0,1]."""
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)


def parse_annotation(xml_path, fine=False):
    """Parse one VOC xml -> (list of 'cls cx cy w h' strings, image basename).

    Skips objects whose name isn't a known GTSDB class rather than guessing.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    filename = root.find("filename").text
    stem = Path(filename).stem

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        fid = FINE_INDEX.get(name)
        if fid is None:
            continue
        cls = fid if fine else superclass_of(fid)
        b = obj.find("bndbox")
        cx, cy, bw, bh = _voc_to_yolo(
            float(b.find("xmin").text), float(b.find("ymin").text),
            float(b.find("xmax").text), float(b.find("ymax").text), w, h)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, stem


def convert(fine=False, val_frac=0.15, seed=0):
    ann_dir = VOC_ROOT / "Annotations"
    img_dir = VOC_ROOT / "JPEGImages"
    if not ann_dir.exists():
        print(f"ERROR: GTSDB VOC not found at {VOC_ROOT}")
        print("Download it first (see module docstring).")
        return False

    names = FINE_NAMES if fine else SUPERCLASS_NAMES
    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        print(f"ERROR: no annotations under {ann_dir}")
        return False

    # Fresh output tree
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val"):
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    kept = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    empty = 0

    for xml_path in xmls:
        lines, stem = parse_annotation(xml_path, fine=fine)
        if not lines:
            empty += 1
            continue
        img_src = img_dir / f"{stem}.jpg"
        if not img_src.exists():
            # some VOC dumps keep .ppm; try that
            alt = img_dir / f"{stem}.ppm"
            if alt.exists():
                img_src = alt
            else:
                continue
        split = "val" if rng.random() < val_frac else "train"
        shutil.copy2(img_src, OUT_ROOT / split / "images" / img_src.name)
        with open(OUT_ROOT / split / "labels" / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))
        kept[split] += 1
        boxes[split] += len(lines)

    # dataset.yaml for ultralytics
    yaml_lines = [
        f"path: {OUT_ROOT}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (OUT_ROOT / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"{'='*60}")
    print(f"GTSDB -> YOLO  ({'43 fine' if fine else '4 super'} classes)")
    print(f"{'='*60}")
    print(f"  train : {kept['train']:4d} images, {boxes['train']:4d} boxes")
    print(f"  val   : {kept['val']:4d} images, {boxes['val']:4d} boxes")
    print(f"  skipped (no known sign): {empty}")
    print(f"  dataset.yaml: {OUT_ROOT / 'dataset.yaml'}")
    return kept["train"] > 0 and kept["val"] > 0


def _selftest():
    # VOC->YOLO maths on a known box in a 1000x500 image
    cx, cy, bw, bh = _voc_to_yolo(100, 50, 300, 250, 1000, 500)
    assert abs(cx - 0.2) < 1e-9 and abs(cy - 0.3) < 1e-9, (cx, cy)
    assert abs(bw - 0.2) < 1e-9 and abs(bh - 0.4) < 1e-9, (bw, bh)
    # out-of-frame corners clamp into [0,1]
    cx, cy, bw, bh = _voc_to_yolo(-10, -10, 1100, 600, 1000, 500)
    assert all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh))
    # superclass mapping covers all 43 ids and only emits 0..3
    got = {superclass_of(i) for i in range(43)}
    assert got == {0, 1, 2, 3}, got
    # spot-check known members
    assert superclass_of(FINE_INDEX["Stop"]) == 3
    assert superclass_of(FINE_INDEX["Speed limit (50km/h)"]) == 0
    assert superclass_of(FINE_INDEX["Turn right ahead"]) == 1
    assert superclass_of(FINE_INDEX["Road work"]) == 2
    print("selftest OK: VOC->YOLO maths, clamping, and 43->4 mapping all correct")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GTSDB VOC -> YOLO detection format")
    p.add_argument("--classes", choices=["4", "43"], default="4",
                   help="4 super-classes (trustworthy) or 43 fine classes")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        _selftest()
    else:
        convert(fine=(a.classes == "43"), val_frac=a.val_frac)
