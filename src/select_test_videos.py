"""Select a small, diverse set of test clips that actually stress the pipeline.

The point is not "grab 15 random clips" — it is to cover the cases the README
flags as hard (night, dense urban, occlusion, small distant lights/signs) so a
before/after comparison exercises the failure modes, not just the easy highway.

Each clip gets objective, CPU-computable descriptors:
  brightness   mean luma           -> day vs night / tunnel
  contrast     luma std            -> washed-out vs crisp
  motion       mean frame-diff     -> ego speed / scene change rate
  edge_density Canny edge fraction -> urban clutter vs open road
  vehicles     mean YOLOv8n cars   -> occlusion / traffic density
  lights       mean YOLOv8n TLs    -> whether traffic-signal logic is exercised

Selection then maximises spread: z-score the descriptors, greedily pick the clip
farthest (in feature space) from those already chosen (farthest-point sampling),
after seeding with the extremes (darkest, busiest, most-cluttered). That yields a
set that spans the space instead of clustering on the common case.

ponytail: YOLOv8n on CPU at a low sample rate is the density proxy. Ceiling — it
undercounts tiny distant lights the same way the pipeline does, so "lights=0"
means "none this model can see", not "none present". Upgrade path = score with the
retrained detector once it exists. Runs on CPU by design so it works without a GPU.

Self-check:  python src/select_test_videos.py --selftest
Real run:    python src/select_test_videos.py --n 12
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR

# Where to look for candidate clips. Missing dirs are skipped, not fatal.
SOURCES = [
    PROJECT_ROOT / "BDDA" / "test" / "camera_videos",
    PROJECT_ROOT / "downloads",
    PROJECT_ROOT / "data",
]

SAMPLE_FRAMES = 12          # frames sampled per clip for descriptors
DETECT_EVERY = 3            # run YOLO on every Nth sampled frame (CPU cost)
VEHICLE_CLS = {2, 3, 5, 7}
LIGHT_CLS = 9

DESCRIPTORS = ["brightness", "contrast", "motion", "edge_density",
               "vehicles", "lights"]


def _sample_indices(total, k):
    """Evenly spaced frame indices, avoiding the very first/last frame."""
    if total <= 0:
        return []
    k = min(k, total)
    return [int(x) for x in np.linspace(total * 0.1, total * 0.9, k)]


def describe_clip(path, detector=None):
    """Compute scene descriptors for one clip. Returns dict or None if unreadable."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return None

    idxs = _sample_indices(total, SAMPLE_FRAMES)
    brights, contrasts, motions, edges = [], [], [], []
    veh_counts, light_counts = [], []
    prev_gray = None

    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brights.append(float(gray.mean()))
        contrasts.append(float(gray.std()))
        # Downscale edges/motion so clip resolution doesn't bias the numbers
        small = cv2.resize(gray, (320, 180))
        edges.append(float((cv2.Canny(small, 80, 160) > 0).mean()))
        if prev_gray is not None:
            motions.append(float(np.abs(small.astype(np.int16) - prev_gray).mean()))
        prev_gray = small.astype(np.int16)

        if detector is not None and j % DETECT_EVERY == 0:
            v = l = 0
            for r in detector(frame, conf=0.25, verbose=False, imgsz=640):
                for b in r.boxes:
                    c = int(b.cls[0])
                    if c in VEHICLE_CLS:
                        v += 1
                    elif c == LIGHT_CLS:
                        l += 1
            veh_counts.append(v)
            light_counts.append(l)

    cap.release()
    if not brights:
        return None

    # Reject dead clips: near-black or frozen footage is a broken file, not a
    # hard night scene. A genuine night clip still has contrast from lights.
    mean_b = float(np.mean(brights))
    mean_c = float(np.mean(contrasts))
    if mean_b < 5.0 or mean_c < 3.0:
        return None

    return {
        "path": str(path),
        "name": path.name,
        "frames": total,
        "brightness": round(float(np.mean(brights)), 2),
        "contrast": round(float(np.mean(contrasts)), 2),
        "motion": round(float(np.mean(motions)) if motions else 0.0, 3),
        "edge_density": round(float(np.mean(edges)), 4),
        "vehicles": round(float(np.mean(veh_counts)) if veh_counts else 0.0, 2),
        "lights": round(float(np.mean(light_counts)) if light_counts else 0.0, 2),
    }


def _zscore(rows):
    """Z-score the descriptor columns. Returns (matrix, feature_names)."""
    mat = np.array([[r[k] for k in DESCRIPTORS] for r in rows], dtype=np.float64)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd, DESCRIPTORS


def select_diverse(rows, n):
    """Farthest-point sampling over z-scored descriptors.

    Seeds with the genuinely hard extremes (darkest, most vehicles, most edges),
    then repeatedly adds the clip maximally far from the current selection. This
    spans the feature space rather than clustering on the average highway clip.
    """
    if len(rows) <= n:
        return list(range(len(rows)))

    Z, feats = _zscore(rows)
    fidx = {f: i for i, f in enumerate(feats)}

    seeds = {
        int(np.argmin(Z[:, fidx["brightness"]])),    # darkest / night
        int(np.argmax(Z[:, fidx["vehicles"]])),      # densest traffic
        int(np.argmax(Z[:, fidx["edge_density"]])),  # most cluttered / urban
        int(np.argmax(Z[:, fidx["lights"]])),        # most traffic signals
    }
    chosen = list(seeds)

    while len(chosen) < n:
        # distance from every clip to its nearest already-chosen clip
        d = np.min(
            np.linalg.norm(Z[:, None, :] - Z[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    return chosen


def gather_clips(sources, limit_per_source=None):
    clips = []
    for src in sources:
        if not src.exists():
            continue
        vids = sorted(src.glob("*.mp4"))
        if limit_per_source:
            vids = vids[:limit_per_source]
        clips.extend(vids)
    return clips


def run(n=12, use_detector=True, limit=None):
    clips = gather_clips(SOURCES, limit_per_source=limit)
    if not clips:
        print("No .mp4 clips found under:", *[str(s) for s in SOURCES], sep="\n  ")
        return

    detector = None
    if use_detector:
        try:
            from ultralytics import YOLO
            wp = MODELS_DIR / "yolov8n.pt"
            if not wp.exists():
                wp = Path(__file__).parent / "yolov8n.pt"
            detector = YOLO(str(wp))
            print(f"Object density via {wp.name} (CPU ok)")
        except Exception as e:
            print(f"Detector unavailable ({e}); scoring on scene stats only.")

    print(f"Profiling {len(clips)} clips ({SAMPLE_FRAMES} frames each)...")
    rows = []
    for i, c in enumerate(clips, 1):
        d = describe_clip(c, detector)
        if d:
            rows.append(d)
        if i % 25 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)}")

    if not rows:
        print("No readable clips.")
        return

    picks = select_diverse(rows, n)
    selected = [rows[i] for i in picks]
    # Order the chosen set darkest->brightest so the list reads sensibly
    selected.sort(key=lambda r: r["brightness"])

    out_json = PROJECT_ROOT / "output" / "test_video_manifest.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"selected": selected, "all_profiled": rows}, f, indent=2)

    print(f"\n{'='*70}\n  SELECTED {len(selected)} TEST CLIPS "
          f"(of {len(rows)} profiled)\n{'='*70}")
    hdr = f"  {'name':22s} {'bright':>7s} {'motion':>7s} {'edges':>7s} {'veh':>5s} {'lights':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in selected:
        tag = "night" if r["brightness"] < 70 else ("urban" if r["edge_density"] > 0.09 else "")
        print(f"  {r['name'][:22]:22s} {r['brightness']:7.1f} {r['motion']:7.2f} "
              f"{r['edge_density']:7.3f} {r['vehicles']:5.1f} {r['lights']:6.1f}  {tag}")
    print(f"\n  Manifest: {out_json}")
    print("  Feed these to: python src/validate_pipeline.py --input <path>")


# --------------------------------------------------------------------------
# Self-check: the selection logic must actually spread across the feature space,
# not just return the first n rows. Build synthetic clusters and assert the
# picker takes from every cluster.
def _selftest():
    rng = np.random.default_rng(0)
    rows = []
    # three tight clusters far apart in feature space
    centres = {
        "night":  dict(brightness=30, contrast=20, motion=2, edge_density=0.05, vehicles=1, lights=0),
        "urban":  dict(brightness=120, contrast=60, motion=8, edge_density=0.18, vehicles=12, lights=4),
        "highway":dict(brightness=160, contrast=40, motion=15, edge_density=0.04, vehicles=3, lights=0),
    }
    for label, c in centres.items():
        for k in range(10):
            r = {"path": f"{label}{k}", "name": f"{label}{k}", "frames": 100}
            for key, base in c.items():
                r[key] = base + float(rng.normal(0, abs(base) * 0.03 + 0.01))
            rows.append(r)

    picks = select_diverse(rows, 6)
    labels = {rows[i]["name"][:-1] if rows[i]["name"][-1].isdigit() else rows[i]["name"]
              for i in picks}
    labels = {"".join(ch for ch in rows[i]["name"] if not ch.isdigit()) for i in picks}
    assert labels == {"night", "urban", "highway"}, \
        f"selection failed to span clusters, got {labels}"
    assert len(set(picks)) == 6, "duplicate picks"
    print("selftest OK: selection spans all 3 clusters with no duplicates")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Select diverse test clips")
    p.add_argument("--n", type=int, default=12, help="how many clips to select")
    p.add_argument("--no-detector", action="store_true",
                   help="skip YOLO object density (faster, scene stats only)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap clips per source dir (for a quick trial)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        _selftest()
    else:
        run(n=a.n, use_detector=not a.no_detector, limit=a.limit)
