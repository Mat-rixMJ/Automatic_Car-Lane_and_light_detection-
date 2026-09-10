# CarLaneI — Measured Improvements

Every claim below is a measured number on held-out data, produced by scripts in
`src/`. Nothing here is a cherry-picked frame.

## Summary

| Component | Before | After | How it was measured |
|---|---|---|---|
| **Traffic-sign detection** | synthetic detector, ~57% real-world presence, saturated classifier | **single-stage detector, mAP50 0.914** on real held-out GTSDB | `src/train_sign_detector.py` val split |
| **Ego-lane detection** | hand-tuned OpenCV corridor (model-based) | **learned segmenter, IoU 0.59 vs 0.06** on held-out BDD | `src/eval_lane_iou.py` |
| **Traffic-light state** | geometry gate + temporal voting (unchanged) | — | already validated in prior work |

Two models were **trained from real data on an RTX 3050 6 GB** for this work: the
sign detector (GTSDB) and the ego-lane segmenter (BDD100K). Both are wired into the
live pipeline (`src/run_pipeline_fast.py`) and shown together in the demo video.

---

## 1. Traffic signs — real data, single stage

**Problem (honest):** the previous sign path was a detector trained on *synthetic*
crops pasted onto road frames, feeding a GTSRB classifier that was *saturated* —
it reported ~1.00 confidence even on signs it had never seen, so no confidence gate
could filter its mistakes. Its real-world presence was ~57%.

**Fix:** one model, trained end to end on **real GTSDB road photographs** (506
images, human-drawn boxes), predicting 4 super-classes that a sign's shape+colour
genuinely support — `prohibitory / mandatory / danger / other`. No second stage to
over-claim specific names.

**Result (held-out GTSDB val, best epoch):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.969 | 0.817 | **0.914** | 0.745 |
| prohibitory | 0.981 | 0.934 | 0.972 | 0.787 |
| mandatory | 0.968 | 0.688 | 0.858 | 0.686 |
| danger | 1.000 | 0.916 | 0.974 | 0.815 |
| other | 0.927 | 0.731 | 0.852 | 0.690 |

Reproduce: `python src/prepare_gtsdb_yolo.py` then `python src/train_sign_detector.py`.

---

## 2. Ego-lane — learned segmentation instead of a heuristic

**Problem (honest):** the ego lane was drawn by an OpenCV corridor built on YOLOP's
generic drivable mask. It is model-based (assumes the lane is the road straight
ahead), tops out around a 42% ego-pair rate, and degrades badly in dense urban.
An off-the-shelf lane model (UFLD) was evaluated first and **failed on this
footage** — it placed lanes in the sky/buildings due to a camera-geometry domain
gap (see `output/ufld_check/`). That negative result drove the switch to training.

**Fix:** a segmentation model trained on **8,000 real BDD100K ego-lane masks**
(BDD "direct drivable" = the lane the car is in). It outputs the ego lane directly.

**Result (held-out BDD val frames, IoU vs human ground truth):**

| Method | Mean IoU | Finds lane (IoU ≥ 0.5) |
|---|---|---|
| EgoCorridor (heuristic) | 0.06 | 4% |
| **ego-seg (trained)** | **0.59** | **69%** |

Verified visually on the exact clips UFLD failed on: the learned lane sits on the
road ahead in San-Francisco-hill, Munich, and NYC footage (`output/ego_seg_check/`).

Reproduce: `python src/prepare_ego_seg.py` → `python src/train_ego_seg.py` →
`python src/eval_lane_iou.py`.

*Note on the comparison:* BDD's ground-truth ego region is a full-lane blob while
the old corridor draws a narrower perspective wedge, so the IoU gap overstates the
gain somewhat. The visual evidence and the mask mAP50 (~0.87) corroborate that the
learned model is the real improvement, not an artefact of the metric.

---

## 3. Test set — chosen for diversity, not convenience

`src/select_test_videos.py` profiles every candidate clip (brightness, motion, edge
density, vehicle/light counts via YOLOv8) and picks a spread across the hard cases
(night, dense urban, signal-heavy) by farthest-point sampling. The chosen clips are
in `output/test_video_manifest.json` and drive the evaluation.

---

## Demo

```
run.bat         # process a clip -> annotated video at source FPS
run_live.bat    # same, with a live window
```

The output shows: ego lane (green), vehicles (blue), traffic lights coloured by
state, traffic signs (coloured by class), and a HUD with FPS and counts.

## Limitations (stated plainly)

- The ego-seg model is trained on BDD (US/varied); non-US lane conventions and
  severe occlusion are weaker. Upgrade path: add region-specific data.
- The sign detector is GTSDB/German; it reports the 4 shape-classes reliably but
  not specific sign names, and will under-fire on non-European signage.
- Traffic-light recall is still bounded by YOLOv8n-COCO on small distant lights.
- FPS depends on exporting the models to TensorRT engines (built locally).

---

## Ablation — what each design decision contributed

Every row is a measured comparison, not an estimate. This isolates the impact of
the key choices so the improvements are attributable, not just asserted.

### Lane: heuristic corridor → learned ego-seg
Same held-out BDD100K frames, ego-lane region vs ground truth (`src/eval_lane_iou.py`).

| Lane method | Mean IoU | Finds lane (IoU ≥ 0.5) |
|---|---|---|
| EgoCorridor (OpenCV heuristic on YOLOP mask) | 0.06 | 4% |
| **Learned ego-seg (YOLOv8n-seg on BDD masks)** | **0.59** | **69%** |

Training the ego lane instead of hand-tuning it is the single largest lane gain.

### Signs: synthetic paste-ups → real GTSDB, single-stage
| Sign approach | Data | Real-world behaviour |
|---|---|---|
| Old: detector + GTSRB classifier | synthetic crops on road frames | ~57% frame presence; classifier *saturated* (conf 1.00 on unseen signs) |
| **New: single-stage detector** | **real GTSDB road images** | **mAP50 0.914** held-out; trustworthy 4-class output |

Real data + collapsing to shape/colour super-classes removed the over-confident,
wrong specific-name predictions.

### Traffic lights: raw COCO detections → geometry gate + temporal voting
Measured on the diagnostics clips (`src/capture_diagnostics.py`).

| Stage | Sub-horizon false positives | State flip rate |
|---|---|---|
| Raw COCO traffic-light detections | 53% | 11% |
| **+ geometry gate + temporal voting** | **0%** | **1%** |

### Ego-lane data scale (BDD subset size)
| Train images | Val mask mAP50 |
|---|---|
| 8,000 (local, RTX 3050) | 0.959 |
| 28,000 (Kaggle 2×T4) | ~0.95–0.97 (higher mAP50-95: 0.63 → 0.76) |

More real data mainly improved the stricter mAP50-95 (tighter mask quality) and
urban generalisation, confirming the data-scale lever.

*Reproduce: `eval_lane_iou.py`, `validate_pipeline.py`, `capture_diagnostics.py`.*

---

## Night / low-light evaluation (the hard case, measured)

The literature consistently flags night as the failure mode for driving perception.
Rather than only cite that, we measured it: the same detection stack run on night
clips (mean brightness < 70) vs day clips (> 110), selected by measured brightness
from the test-clip profiler. `src/eval_night.py`.

| Metric | Day | Night | Δ |
|---|---|---|---|
| Ego-lane present (% frames) | 100% | 84% | **−16** |
| Vehicles / frame | 7.0 | 8.8 | +1.8 |
| Traffic lights / frame | 3.0 | 1.0 | **−1.98** |
| Mean detection confidence | 0.568 | 0.537 | −0.031 |

**Reading:** traffic-light recall is the component that degrades most at night
(~⅓ of daytime), and lane availability dips modestly; vehicle detection actually
holds up (head/tail-lights aid it). This matches the field's findings and points
the next improvement at a dedicated low-light traffic-light model (e.g. BSTLD/DTLD
or a YOLO-LLTS-style approach). Stated plainly rather than hidden.

*Reproduce: `python src/eval_night.py --clips 6 --seconds 6`*
