# Automatic Car Lane & Traffic Light Detection

Real-time lane and traffic-signal perception for autonomous driving, built with
computer vision and CNNs. Runs at **28–32 FPS on 720p** on a laptop RTX 3050 6 GB.

Scope is deliberately narrow: **ego-lane detection** and **traffic-light detection
with state classification**, plus vehicle detection. Traffic-sign recognition was
built, evaluated, and then removed from the pipeline — see
[Traffic signs](#traffic-signs-removed-and-why) for the honest reason.

---

## Demo

```bash
# Edit the VIDEO path inside run.bat, then:
run.bat
```

or directly:

```bash
python src/run_pipeline_fast.py --input path/to/video.mp4 --live --no-record --no-crop
```

| Flag | Meaning |
|------|---------|
| `--live` | Show a live OpenCV window |
| `--no-record` | Don't write an output file (fastest) |
| `--output PATH` | Save the annotated video |
| `--no-crop` | Don't auto-crop ultra-wide input |
| `--display-h 720` | Shrink the preview window (helps at 1080p) |

Press `q` or `Esc` to stop.

---

## What's on screen

| Overlay | Meaning |
|---------|---------|
| Green corridor + cyan edges | Detected **ego lane** (your lane only, not the whole road) |
| Pale green speckle | Raw YOLOP lane-marking pixels |
| Orange boxes | Vehicles — car / motorcycle / bus / truck |
| Red / Yellow / Green boxes | Traffic lights, coloured by classified state |
| `LANE DEPARTURE` / `drifting` | Lane-departure warning from ego offset |
| Top-left HUD | FPS, vehicle count, light count |

---

## Architecture

```
                     Input frame (dashcam video)
                                |
              +-----------------+------------------+
              v                                    v
     +------------------+                 +--------------------+
     |  YOLOP  (CNN)    |                 |  YOLOv8n  (CNN)    |
     |  TensorRT FP/TF32|                 |  TensorRT          |
     |                  |                 |                    |
     |  drivable area   |                 |  vehicles          |
     |  lane pixels     |                 |  traffic lights    |
     +--------+---------+                 +---------+----------+
              |                                     |
              v                                     v
     +------------------+                 +--------------------+
     |  EgoCorridor     |                 | TrafficLightTracker|
     |  (OpenCV)        |                 |  (OpenCV)          |
     |                  |                 |                    |
     | occlusion bridge |                 | geometry gate      |
     | scanline spans   |                 | colour opponency   |
     | perspective prior|                 | temporal voting    |
     | temporal EMA     |                 |                    |
     | -> ego offset    |                 | -> stable R/Y/G    |
     +--------+---------+                 +---------+----------+
              |                                     |
              +------------------+------------------+
                                 v
                    Annotated frame + LDW alert
```

Two neural networks, both TensorRT-compiled. YOLOP runs every 4th frame (road
geometry changes slowly), YOLOv8 every 3rd, with results cached between — this is
what buys the frame rate.

### Tech stack

- **CNN / deep learning** — YOLOP (lane + drivable-area segmentation), YOLOv8n (detection)
- **Computer vision / OpenCV** — corridor construction, morphological occlusion
  bridging, scanline road analysis, perspective priors, colour-opponency light
  classification, temporal smoothing
- **PyTorch** for the models, **TensorRT** for inference (~2.7× faster than PyTorch on YOLOP)

---

## Results

Measured with `src/capture_diagnostics.py`, which runs detection on **every frame**
(no skipping) and reports metrics rather than impressions.

### Detection coverage

| | Frankfurt (city) | BDDA 100 (highway) | BDDA 1003 (urban) |
|---|---|---|---|
| Ego corridor available | 100% | 100% | 100% |
| Vehicles | 10.5/frame | 8.8/frame | 11.6/frame |
| Traffic-light state classified | 87% | 96% | 98% |

### Traffic-light filtering

Raw COCO traffic-light detections are noisy. A geometry gate plus temporal state
voting cleans them up:

| | Before | After |
|---|---|---|
| Detections below the horizon | **53%** | **0%** |
| State flipping between frames | **11%** | **1%** |

Anything below the horizon is a taillight, reflection or sign — a real traffic
light hangs above the roadway.

### Ego-lane corridor stability

| | Frankfurt | BDDA 100 | BDDA 1003 |
|---|---|---|---|
| Centre jitter, before | 34 px/frame | 62 px/frame | 60 px/frame |
| Centre jitter, after | **9.8 px** | **3.0 px** | 30 px |
| Corridor width, before | 47% of frame | 66% | 48% |
| Corridor width, after | **27%** | **42%** | 13% |

### Performance (RTX 3050 6 GB laptop, 1280×720)

| Stage | Cost/frame |
|---|---|
| YOLOP (TensorRT) | 25.0 ms |
| YOLOv8n (TensorRT) | 24.9 ms |
| Corridor + light tracking | 10.7 ms |
| Display | 7.1 ms |
| Read + draw | 2.9 ms |

**28–32 FPS** live at 720p with frame skipping. Exporting YOLOP to TensorRT alone
took it from 67 ms to 25 ms.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Build TensorRT engines (one-off, a few minutes each)
python src/export_yolop_trt.py     # -> models/yolop_384.engine
python src/export_tensorrt.py      # -> models/yolov8n.engine
```

Requires an NVIDIA GPU with CUDA. Without the `.engine` files the pipeline falls
back to PyTorch automatically — correct, just slower.

Model weights and datasets are gitignored; the export scripts fetch and build
what they need.

---

## Project layout

```
src/
  run_pipeline_fast.py     Main pipeline (entry point)
  ego_corridor.py          Ego-lane corridor from the drivable-area mask
  traffic_light.py         Light geometry gate + temporal state voting
  lane_fit.py              Lane-line fitting (connected components)
  trt_runner.py            Minimal TensorRT inference wrapper
  utils.py                 Shared paths and helpers

  export_yolop_trt.py      YOLOP  -> ONNX -> TensorRT engine
  export_tensorrt.py       YOLOv8 -> ONNX -> TensorRT engine

  capture_diagnostics.py   Measures spill / width / jitter / light stability
  validate_pipeline.py     Per-class detection rates with pass-fail gates
  profile_current.py       Per-stage timing breakdown
  measure_lane_width.py    Empirical lane-width study
  test_lane_fit.py         Self-checks for the lane fitter (10/10)

config/                    Thresholds and class maps
run.bat                    One-click launcher
memory.md                  Engineering log: what was measured and why
```

---

## Engineering notes

Design decisions here came from measurement, and several overturned my initial
assumptions. Full detail in [`memory.md`](memory.md); the highlights:

**Lane-line pairing is not viable on this footage.** Across ~2800 frames, only
**4** produced two fitted lane lines that both span a common height range. Dashes,
occlusion, single-sided markings and intersections mean the ego lane's two
boundaries are almost never both cleanly visible. So the corridor is built from the
drivable-area mask (present in >92% of frames) and *refined* by lane markings
rather than defined by them.

**A coverage metric without a correctness metric is a trap.** An early fallback
drove "ego pair found" from 0% to 99% — by outlining the entire road again, which
was the original bug. The number improved while the output got worse.

**Profile before optimising.** My first speed pass targeted 6.5 ms of drawing code
while YOLOP sat at 67 ms untouched.

**Don't test a component against its own assumptions.** The first lane-fitter test
suite generated synthetic masks using the same perspective model the fitter used.
It passed 8/8 while the fitter was visibly broken on real video. The rewritten
tests assert properties that were actually violated, and one runs real YOLOP masks.

**Vehicles break the road mask.** Taking the longest contiguous road run per
scanline lands on a fragment beside an occluding car, collapsing the corridor to a
sliver. Fixed with a horizontal morphological close plus anchor-based run
selection.

---

## Traffic signs: removed, and why

A German sign detector was trained (YOLOv8n on GTSDB-style data, 99.3% mAP on its
validation set) and paired with a GTSRB CNN classifier (96.81% on the official
GTSRB test set). Both were removed from the pipeline.

The classifier cannot be trusted for specific sign names, and it cannot tell you
so. GTSRB has 43 classes and **no U-turn sign**, so a U-turn is out-of-distribution
— yet across 314 real detected crops the classifier reported **median confidence
1.00 with a median top-1 vs top-2 margin of 1.00**. It labelled a real U-turn sign
"Ahead only" at full confidence. A confidence or margin gate cannot filter that,
because the model is saturated.

Also, the detector's 99.3% mAP was against a **synthetic** training set (GTSRB
crops composited onto road frames), not real road photography. Its real-world rate
was ~57% frame presence.

Honest fix, not yet done: train a detector on genuinely annotated per-class sign
photographs so specific names come from the detector itself. The code remains in
`src/` (`sign_classifier.py`, `train_combined_detector.py`) for that future work.

---

## Known limitations

- **Dense urban footage is the weak case.** On BDDA 1003 the corridor is 13% of
  frame width with 23% off-road spill and 30 px jitter — the road mask is small and
  fragmented, leaving little to lock onto. Frankfurt and highway clips are solid.
- **The corridor is model-based, not detected.** It assumes the ego lane is the
  road surface directly ahead, so it can't represent a lane change in progress and
  lags briefly through one. A model trained to output ego-lane boundaries directly
  (CLRNet, UFLD, or YOLOP fine-tuned on ego-lane labels) is the real upgrade.
- **Traffic-light recall is capped by YOLOv8n-COCO** on small distant lights. A
  dedicated traffic-light model would raise it.
- **Light state uses colour opponency**, so it can struggle at night and with
  strong backlighting.
- **1080p is CPU-bound**, not GPU-bound — video decode dominates. 720p is the
  practical sweet spot.
- Traffic-light association is by grid cell, not IoU tracking, so two lights very
  close together can merge into one track.

---

## References

- **YOLOP** — Wu et al., *You Only Look Once for Panoptic Driving Perception*, MIR 2022
- **YOLOv8** — Ultralytics
- **GTSRB** — Stallkamp et al., IJCNN 2011
- **BDD-A** — Xia et al., *Predicting Driver Attention in Critical Situations*
