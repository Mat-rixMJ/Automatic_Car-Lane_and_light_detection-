"""Download the Indian dashcam sign dataset locally and MERGE it with the already
converted German GTSDB (data/gtsdb_yolo) into ONE 4-super-class YOLO dataset at
data/signs_merged, ready for src/train_sign_detector.py to train on the RTX 3050.

This is the LOCAL twin of kaggle/train_sign_detector_kaggle.py. It reuses the same
Indian class-name -> super-class keyword mapping so the two paths agree. German is
already 4-super (from prepare_gtsdb_yolo.py), so we just copy it in.

  Indian : akbaralibatti/indian-traffic-sign-yolo11  (7.5k real dashcam photos, YOLO)
  German : data/gtsdb_yolo                            (already 4 super-classes)

Run:
  python src/prepare_indian_signs.py            # download (if needed) + merge
  python src/prepare_indian_signs.py --no-download   # merge only (data already local)
  python src/prepare_indian_signs.py --selftest      # mapping check, no I/O

Then:
  python src/train_sign_detector.py --data data/signs_merged/dataset.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT

INDIAN_SLUG = "akbaralibatti/indian-traffic-sign-yolo11"
INDIAN_DL = PROJECT_ROOT / "data" / "indian_signs_dl"
GTSDB_YOLO = PROJECT_ROOT / "data" / "gtsdb_yolo"     # already 4-super
OUT = PROJECT_ROOT / "data" / "signs_merged"

SUPERCLASS_NAMES = ["prohibitory", "mandatory", "danger", "other"]

# ---- Indian class NAME -> super-class by shape/colour keyword (mirror of the
# ---- Kaggle script's indian_super()). First match wins; unmatched -> 'other'.
INDIAN_OVERRIDES = {}

def indian_super(name):
    n = name.lower().strip()
    if n in INDIAN_OVERRIDES:
        return INDIAN_OVERRIDES[n]
    danger_kw = ["warning", "caution", "curve", "bend", "slippery", "narrow",
                 "hump", "bump", "dip", "ripple", "gap", "cattle", "pedestrian",
                 "school", "children", "cross", "round about", "roundabout ahead",
                 "junction", "gap in median", "falling", "cycle cross", "animal",
                 "men at work", "road work", "barrier", "ferry", "ford",
                 "steep", "descent", "ascent", "hairpin", "loose", "guard"]
    mandatory_kw = ["compulsory", "mandatory", "ahead only", "turn right",
                    "turn left", "keep left", "keep right", "go straight",
                    "compulsory ahead", "sound horn", "cycle track",
                    "pedestrian only", "bus", "direction"]
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
    return 3


def download_indian():
    if INDIAN_DL.exists() and any(INDIAN_DL.rglob("*.jpg")):
        print(f"Indian data already present at {INDIAN_DL}")
        return
    print(f"Downloading {INDIAN_SLUG} -> {INDIAN_DL} ...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    INDIAN_DL.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(INDIAN_SLUG, path=str(INDIAN_DL), unzip=True, quiet=False)
    print("  download complete.")


def _find_indian_layout():
    """Return (images_root, labels_root, data_yaml) for the downloaded Indian set."""
    yaml_path = None
    for p in INDIAN_DL.rglob("data.yaml"):
        yaml_path = p
        break
    if not yaml_path:
        return None, None, None
    base = yaml_path.parent
    img_dirs = [q for q in base.rglob("*") if q.is_dir() and q.name == "images"]
    lbl_dirs = [q for q in base.rglob("*") if q.is_dir() and q.name == "labels"]
    img_root = img_dirs[0].parent if img_dirs else base
    lbl_root = lbl_dirs[0].parent if lbl_dirs else base
    return img_root, lbl_root, yaml_path


def merge():
    for split in ("train", "val"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    counts = {i: 0 for i in range(4)}

    # ---- German: already 4-super, just copy with a 'de_' prefix
    de_imgs = 0
    for split in ("train", "val"):
        img_dir = GTSDB_YOLO / split / "images"
        lbl_dir = GTSDB_YOLO / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            shutil.copy2(img, OUT / split / "images" / f"de_{img.name}")
            shutil.copy2(lbl, OUT / split / "labels" / f"de_{img.stem}.txt")
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if line:
                    counts[int(float(line.split()[0]))] = \
                        counts.get(int(float(line.split()[0])), 0) + 1
            de_imgs += 1
    print(f"German copied: {de_imgs} images")

    # ---- Indian: remap class ids via data.yaml names -> super-class
    import yaml as _yaml
    img_root, lbl_root, yaml_path = _find_indian_layout()
    if not yaml_path:
        print("  !! Indian data.yaml not found under", INDIAN_DL)
        print("     download may have failed; run without --no-download")
        return False
    cfg = _yaml.safe_load(Path(yaml_path).read_text())
    raw = cfg.get("names")
    names = [raw[k] for k in sorted(raw, key=lambda x: int(x))] if isinstance(raw, dict) \
        else list(raw)
    remap = {i: indian_super(nm) for i, nm in enumerate(names)}
    print("\n--- Indian class -> superclass mapping (eyeball this) ---")
    for i, nm in enumerate(names):
        print(f"  {i:3d} {nm:40s} -> {SUPERCLASS_NAMES[remap[i]]}")

    # Build a stem->image index ONCE (the extracted tree nests images under
    # train/images/train etc.; a per-label rglob is O(n^2) and re-copies dupes).
    img_index = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in Path(img_root).rglob(ext):
            img_index.setdefault(p.stem, p)

    # Dedupe labels by stem too - the same stem appears in multiple nested label
    # dirs; process each stem once.
    label_by_stem = {}
    for lp in Path(lbl_root).rglob("*.txt"):
        label_by_stem.setdefault(lp.stem, lp)

    import random
    rng = random.Random(0)
    in_imgs = 0
    for stem, lp in label_by_stem.items():
        img = img_index.get(stem)
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
        low = str(lp).lower()
        split = "val" if ("val" in low or "valid" in low or "test" in low
                          or rng.random() < 0.15) else "train"
        shutil.copy2(img, OUT / split / "images" / f"in_{stem}{img.suffix}")
        (OUT / split / "labels" / f"in_{stem}.txt").write_text("\n".join(out_lines))
        in_imgs += 1
    print(f"Indian merged: {in_imgs} images")

    (OUT / "dataset.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(SUPERCLASS_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(SUPERCLASS_NAMES)))
    n_tr = len(list((OUT / "train" / "images").glob("*")))
    n_va = len(list((OUT / "val" / "images").glob("*")))
    print(f"\n{'='*60}")
    print(f"MERGED SIGN DATASET  train={n_tr}  val={n_va}")
    print("boxes/superclass: "
          + ", ".join(f"{SUPERCLASS_NAMES[k]}={counts.get(k,0)}" for k in range(4)))
    print(f"dataset.yaml: {OUT / 'dataset.yaml'}")
    print(f"{'='*60}")
    return n_tr > 0 and n_va > 0


def _selftest():
    checks = {
        "Speed limit 40": "prohibitory", "No Parking": "prohibitory",
        "Compulsory Ahead": "mandatory", "Compulsory Turn Right": "mandatory",
        "Pedestrian Crossing": "danger", "School Ahead": "danger",
        "Cattle": "danger", "Stop": "other", "Give Way": "other",
    }
    bad = []
    for name, expect in checks.items():
        got = SUPERCLASS_NAMES[indian_super(name)]
        if got != expect:
            bad.append((name, expect, got))
    assert not bad, f"mapping mismatches: {bad}"
    print("selftest OK: Indian keyword mapping matches expected super-classes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true",
                    help="skip download, merge whatever is already in data/")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        if not a.no_download:
            download_indian()
        ok = merge()
        print("\nREADY. Train with:\n  python src/train_sign_detector.py "
              "--data data/signs_merged/dataset.yaml" if ok else "\nMERGE FAILED.")
