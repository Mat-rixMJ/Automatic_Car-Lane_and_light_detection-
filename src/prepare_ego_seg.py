"""Convert BDD100K drivable-area masks into a YOLOv8-seg dataset for the EGO LANE.

The pipeline's lane output is an ego-lane corridor. Today that corridor is
hand-built from YOLOP's drivable-area mask with OpenCV heuristics — the README
measures it topping out (42% ego-pair rate, urban failure, 30px jitter). BDD100K
ships 70k human-annotated drivable masks where:

    class 0 = DIRECT drivable  = the lane the car is in  (THE EGO LANE)
    class 1 = ALTERNATIVE       = other lanes
    class 2 = background

So "direct drivable" is exactly the ego lane, already labelled by humans. We train
a model to segment it directly — a learned ego-lane region instead of a heuristic
one. Verified visually (output/mask_check): class 0 follows the ego lane round
curves, class 1 is the oncoming/other lane.

This writes YOLOv8-seg format (one polygon per instance, class 0). We take the
largest few contours of the class-0 region per image so a split lane (occluded by
a car) still yields its pieces.

ponytail: single class "ego". Ceiling — it won't distinguish your lane from the
one you're changing into mid-manoeuvre (BDD labels direct-drivable as one blob).
Upgrade path = BDD's lane-line labels, which this download does not include.

Self-check:  python src/prepare_ego_seg.py --selftest
Build:       python src/prepare_ego_seg.py --n-train 8000 --n-val 1000
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT

BDD = PROJECT_ROOT / "BDD100K"
IMG = BDD / "images"
MASK = BDD / "labels" / "drivable_labels" / "masks"
OUT = PROJECT_ROOT / "data" / "ego_seg"

EGO_VALUE = 0                 # class 0 in the BDD mask = direct drivable = ego
MIN_AREA_FRAC = 0.004         # ignore tiny specks (<0.4% of frame)
MAX_POLYS = 3                 # keep up to N contours (handles occlusion splits)
POLY_EPS_FRAC = 0.004         # contour simplification, fraction of perimeter


def mask_to_polys(mask):
    """class-0 region -> list of normalized YOLO-seg polygons (each flat x,y..).

    Returns [] when there's no usable ego region, so empty frames become
    background (valid in YOLO-seg: image with no label file).
    """
    h, w = mask.shape
    ego = (mask == EGO_VALUE).astype(np.uint8)
    if ego.sum() < MIN_AREA_FRAC * h * w:
        return []
    # Close small holes (lane paint, shadows) so the region is one piece.
    ego = cv2.morphologyEx(ego, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(ego, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:MAX_POLYS]
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA_FRAC * h * w:
            continue
        eps = POLY_EPS_FRAC * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        norm = approx.astype(np.float64)
        norm[:, 0] /= w
        norm[:, 1] /= h
        norm = np.clip(norm, 0.0, 1.0)
        polys.append(norm.reshape(-1).tolist())
    return polys


def convert(n_train, n_val):
    import random
    for split, n in (("train", n_train), ("val", n_val)):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        masks = sorted((MASK / split).glob("*.png"))
        random.Random(0).shuffle(masks)
        masks = masks[:n]
        kept = empty = 0
        for i, mp in enumerate(masks, 1):
            stem = mp.stem
            img_src = IMG / split / f"{stem}.jpg"
            if not img_src.exists():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            polys = mask_to_polys(m)
            # symlink-free copy of the image path via cv2 re-encode is wasteful;
            # instead write a tiny relative reference by copying the file.
            import shutil
            shutil.copy2(img_src, OUT / split / "images" / img_src.name)
            lbl = OUT / split / "labels" / f"{stem}.txt"
            if polys:
                with open(lbl, "w") as f:
                    for p in polys:
                        f.write("0 " + " ".join(f"{v:.6f}" for v in p) + "\n")
                kept += 1
            else:
                lbl.write_text("")          # explicit background
                empty += 1
            if i % 1000 == 0:
                print(f"  {split}: {i}/{len(masks)}")
        print(f"  {split}: {kept} with ego polygon, {empty} background")

    (OUT / "dataset.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\n\nnc: 1\nnames:\n  0: ego_lane\n")
    print(f"  dataset.yaml -> {OUT / 'dataset.yaml'}")


def _selftest():
    # A synthetic mask: a trapezoid of class 0 in the lower-middle. Expect one
    # polygon, normalized in [0,1], round-tripping to roughly the same area.
    h, w = 720, 1280
    m = np.full((h, w), 2, np.uint8)
    pts = np.array([[600, 400], [680, 400], [900, 700], [380, 700]], np.int32)
    cv2.fillPoly(m, [pts], EGO_VALUE)
    polys = mask_to_polys(m)
    assert len(polys) == 1, f"expected 1 poly, got {len(polys)}"
    arr = np.array(polys[0]).reshape(-1, 2)
    assert arr.shape[1] == 2 and len(arr) >= 3
    assert arr.min() >= 0.0 and arr.max() <= 1.0, "polygon not normalized"
    # de-normalize and compare area to the source (within 15%)
    dn = arr.copy(); dn[:, 0] *= w; dn[:, 1] *= h
    a_out = cv2.contourArea(dn.astype(np.float32))
    a_in = cv2.contourArea(pts.astype(np.float32))
    assert abs(a_out - a_in) / a_in < 0.15, f"area off: {a_out:.0f} vs {a_in:.0f}"
    # a near-empty mask -> no polygon (becomes background)
    m2 = np.full((h, w), 2, np.uint8)
    m2[:5, :5] = EGO_VALUE
    assert mask_to_polys(m2) == [], "tiny speck should be dropped"
    print("selftest OK: polygon extraction, normalization, area, background")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BDD drivable -> YOLO-seg ego lane")
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
    else:
        convert(a.n_train, a.n_val)
