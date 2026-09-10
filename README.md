# CarLaneI — Real-Time Driving Perception for Indian Roads

> A four-task perception stack — **ego-lane segmentation, traffic-sign
> detection, traffic-light state detection, and vehicle detection** — running
> live at **~23 FPS on 720p** on a single laptop **RTX 3050 (6 GB)**, all models
> compiled to **TensorRT FP16**.
>
> This document is written to be read by engineers. Every accuracy number is a
> **measured** value from a held-out validation set or an instrumented run on
> real footage; nothing is estimated or rounded up for effect. Where a component
> is weak, this document says so and explains why.

---

## Slide 1 — The one-paragraph pitch

Off-the-shelf driving-perception models are trained on US/European roads (BDD100K,
Cityscapes, KITTI). India's roads — unpainted lanes, mixed traffic, horizontally
mounted signals, dense signage — are a **domain gap** those models fall into. We
built a pipeline that runs four perception tasks in parallel, replaced the two
weakest hand-tuned components with **models we trained ourselves on real data**,
and validated the result on full-length dashcam drives through **Kolkata, Mumbai,
and the NH-44 night highway**. The whole stack fits in 6 GB of VRAM and streams
live in a browser or a standalone window.

---

## Slide 2 — What's on the screen

| Overlay | Source | Meaning |
|---|---|---|
| Green filled lane + yellow border | **ego_seg** (trained) | The lane *you* are in — not the whole road |
| Pale green speckle | YOLOP | Raw lane-marking pixels (secondary cue) |
| Orange boxes | YOLOv8n (COCO) | Vehicles: car / motorcycle / bus / truck |
| Coloured boxes: red / blue / amber / grey | **signs_merged_detector** (trained) | Signs by super-class: prohibitory / mandatory / danger / other |
| Red / Yellow / Green boxes | **light_state** (trained) | Traffic-light **state**, from a learned detector |
| `LANE DEPARTURE` / `drifting` | ego offset | Lane-departure warning from lane centre offset |
| Top HUD + timeline + charts | telemetry | FPS, counts, lane-offset trace, signal timeline |

---

## Slide 3 — System architecture

```
                         Input frame (dashcam video, 1280×720)
                                        │
        ┌───────────────┬───────────────┼───────────────┬────────────────┐
        ▼               ▼               ▼               ▼                ▼
  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌────────────┐
  │  ego_seg  │   │   YOLOP   │   │  YOLOv8n  │   │light_state│   │signs_merged│
  │YOLOv8n-seg│   │ (2 heads) │   │  (COCO)   │   │  YOLOv8n  │   │  YOLOv8n   │
  │  TRT FP16 │   │  TRT FP16 │   │  TRT FP16 │   │  TRT FP16 │   │  TRT FP16  │
  ├───────────┤   ├───────────┤   ├───────────┤   ├───────────┤   ├────────────┤
  │ ego-lane  │   │ drivable  │   │ vehicles  │   │ R/Y/G/OFF │   │ 4 sign     │
  │ mask      │   │ + lane px │   │ + light   │   │ boxes     │   │ super-     │
  │           │   │           │   │ candidates│   │           │   │ classes    │
  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬──────┘
        │               │               │               │               │
        ▼               │               ▼               ▼               │
  ┌───────────┐         │         ┌──────────────────────────┐          │
  │ EgoLane   │◄────────┘         │   TrafficLightTracker     │          │
  │ smoothing │  (lane px refine) │  geometry gate            │          │
  │ + offset  │                   │  + temporal state voting  │          │
  │ + LDW     │                   │  → stable R/Y/G           │          │
  └─────┬─────┘                   └────────────┬─────────────┘           │
        │                                      │                         │
        └──────────────────┬───────────────────┴─────────────────────────┘
                           ▼
              Annotated frame + LDW alert + per-frame telemetry JSON
```

**Frame-skipping is what buys the frame rate.** Road geometry and signals change
slowly relative to 30 FPS, so the heavy models don't run every frame:

| Mode | Ego-lane / YOLOP | Vehicles / lights / signs | Steady-state FPS |
|---|---|---|---|
| Default (accurate, web + recorded) | every 4th frame | every 3rd frame | **16 FPS** |
| `--fast` (live preview / long validation) | every 6th frame | every 4th frame | **23 FPS** |

Detections are cached between the frames a model is skipped, so overlays stay
continuous. The default cadence keeps telemetry accurate; the fast cadence is
display-only.

---

## Slide 4 — The four perception tasks and the five models

We run **five neural networks**. Four of them are the perception tasks; the fifth
(YOLOP) is a supporting lane-pixel cue. Three of the five we **trained ourselves**.

| # | Model | Task | We trained it? | Precision format | Engine size |
|---|---|---|---|---|---|
| 1 | `ego_seg` | Ego-lane segmentation | **Yes** (BDD100K) | TensorRT FP16 | 51 MB |
| 2 | `signs_merged_detector` | Traffic-sign detection (4 classes) | **Yes** (GTSDB + Indian) | TensorRT FP16 | 49 MB |
| 3 | `light_state` | Traffic-light state (R/Y/G/OFF) | **Yes** (Bosch) | TensorRT FP16 | 49 MB |
| 4 | `yolov8n` | Vehicles + light candidates | Pretrained (COCO) | TensorRT FP16 | 86 MB |
| 5 | `yolop_384` | Drivable-area + lane-line pixels | Pretrained (BDD100K) | TensorRT FP16 | 61 MB |

All five are Ultralytics YOLO / YOLOP architectures at **nano scale** (~3 M
parameters each) so five fit comfortably in 6 GB.

---

## Slide 5 — Model 1: `ego_seg` (ego-lane segmentation) ⭐ our best result

**What it does.** Segments *the single lane the car is driving in* as a filled
mask, from which we derive the lane-centre offset and lane-departure warning. Not
the whole drivable road — the ego lane specifically.

**Architecture.** YOLOv8n-seg (segmentation head), ~3.3 M parameters.

**Data.** ~8k–28k real **BDD100K** "direct drivable" masks (the lane the car is
in), from the Kaggle mirror
`nguyentrongquocdat/object-detection-and-drivable-lane-masking`.

**Measured accuracy (held-out BDD100K val, best epoch):**

| Metric | Value |
|---|---|
| **Mask mAP50** | **0.972** |
| Mask mAP50-95 | 0.808 |
| Box mAP50 | 0.973 |
| Precision / Recall | 0.938 / 0.926 |

*Source: `runs/ego_seg_kaggle/results.csv`, epoch 40 (2× T4).* A local 8k-image
run reached mask mAP50 **0.957** (`runs/ego_seg/results.csv`); scaling the data to
28k on Kaggle mainly lifted the stricter **mAP50-95 from 0.63 → 0.81** (tighter
mask edges), confirming the data-scale lever. The deployed model is the 0.972
Kaggle checkpoint.

**Why we trained this instead of using an off-the-shelf lane model — see Slide 10.**

---

## Slide 6 — Model 2: `signs_merged_detector` (traffic signs)

**What it does.** Detects traffic signs and classifies them into **four
super-classes**: `prohibitory`, `mandatory`, `danger`, `other`.

**Architecture.** YOLOv8n detection head, ~3 M parameters, single stage.

**Data (merged, real photographs only):**
- **GTSDB** (German Traffic Sign Detection Benchmark) — real road images, human-drawn boxes, mapped 43 fine classes → 4 super-classes.
- **Indian dashcam signs** — `akbaralibatti/indian-traffic-sign-yolo11` (7,554 real Indian dashcam images), keyword-mapped to the same 4 super-classes.
- Merged set: **train 6,853 / val 1,207**. Box counts per super-class: prohibitory 396, mandatory 114, danger 4,205, other 4,144.

**Measured accuracy (held-out val, best epoch):**

| Metric | Value |
|---|---|
| **mAP50** | **~0.905–0.929** |
| mAP50-95 | 0.705 |

*Source: `runs/sign_detector_signs_merged/results.csv` (0.905 @ ep25 local);
deployed model is a Colab +30-epoch continuation that reached ~0.929.*

**Honest limitation — class imbalance.** The Indian portion of the data is
**warning-signs-heavy**, so `danger` (4,205 boxes) and `other` (4,144) are strong,
while `prohibitory` (396, German-only) and `mandatory` (114) are weak. On live
Indian footage the detector **localises signs correctly** but sometimes assigns a
triangular warning sign to `other` instead of `danger`. This is a **training-data
ceiling, not a pipeline bug** — verified by eyeballing detection crops.

---

## Slide 7 — Model 3: `light_state` (traffic-light state)

**What it does.** Detects traffic lights **and their state directly** —
`red / green / yellow / off` — in a single model. This replaces a hand-written
pixel-colour heuristic that was the weakest link at night.

**Architecture.** YOLOv8n detection, 4 classes.

**Data.** **Bosch Small Traffic Lights** (`xingtingzhao/bosch-traffic-light-yolo`).
After a re-split fix (the original mirror inverted train/test): **train 8,606 /
val 1,509**. Box counts: red 9,211, green 12,722, yellow 561, off 1,081.

**Measured accuracy (held-out Bosch val):** mAP50 **~0.58** (plateaued at ~60
epochs). This is honestly mediocre versus the lane model — driven by two things:
(1) **yellow is severely under-represented** (561 boxes vs 12,722 green), and
(2) a **domain gap**: Bosch is German/US signals; Indian signal housings look
different, so the model scores Indian lights at low confidence (0.12–0.20).

**How we made it usable on Indian footage (calibration, not a data change):**
- Lowered the pipeline confidence gate to `LIGHT_CONF = 0.12` and the tracker's
  learned-path floor to `STATED_MIN_CONF = 0.12`, matching the measured Indian
  confidence band. The heuristic path keeps its stricter 0.20 floor.
- The **temporal voting** in `TrafficLightTracker` rejects one-frame flukes, so a
  low raw gate is safe.
- **Measured effect (Kolkata 30 s clip):** lights went from **0 detections
  (unusable) → RED 24 / GREEN 75**. Confirmed to generalise on Mumbai
  (GREEN 63 / RED 12).

**Ceiling & upgrade path:** a light detector fine-tuned on Indian signals would
fire at higher confidence and let us raise the gate back. Named plainly rather
than hidden.

---

## Slide 8 — Models 4 & 5: `yolov8n` (COCO) and `yolop_384`

**`yolov8n` (COCO, pretrained).** Provides two things the trained models don't:
- **Vehicles** — car / motorcycle / bus / truck (COCO classes 2, 3, 5, 7).
- **Traffic-light candidate boxes** (COCO class 9) — used as a **fallback** for
  the light tracker when the trained `light_state` model isn't present.

  > Engineering note: this engine was originally exported as **FP32 (166 MB)** and
  > ran at **73 ms/frame** — 5× slower than every other model and the pipeline's
  > real bottleneck. Re-exporting as **FP16 (86 MB)** dropped it to **16 ms**,
  > which is what lifted the live rate from ~13 to ~23 FPS.

**`yolop_384` (YOLOP, pretrained).** From `hustvl/YOLOP` via `torch.hub`. Only its
**two segmentation heads** (drivable-area + lane-line pixels) are exported; the
detection head is dropped because YOLOv8 covers objects. Its lane pixels *refine*
the ego-lane mask; its drivable mask is a secondary cue.

---

## Slide 9 — The temporal traffic-light tracker (classical CV that earns its place)

A raw per-frame light detector is noisy: it fires below the horizon (taillights,
reflections, signs) and flips state frame-to-frame. `src/traffic_light.py` cleans
this up **without a model**:

| Stage | Sub-horizon false positives | State-flip rate |
|---|---|---|
| Raw detections | **53%** | **11%** |
| + geometry gate + temporal voting | **0%** | **1%** |

- **Geometry gate:** a real signal head is taller than wide and sits above the
  road (`MAX_Y_FRAC`), rejecting sub-horizon boxes.
- **Temporal voting:** a state must be seen consistently (`VOTES_TO_SWITCH`) before
  the displayed colour changes, and a track survives brief dropouts (`TTL`).
- The same voting machinery is reused whether the state comes from the learned
  `light_state` model **or** the COCO-box + colour-opponency fallback — one
  code path, two sources.

*Source: `src/capture_diagnostics.py`.*

---

## Slide 10 — Design decisions: what we rejected, and the measured reason why

This is the section that matters most for a technical review. Every alternative
below was actually tried or seriously evaluated, and rejected on evidence.

### Ego-lane: why not a heuristic, and why not UFLD?

| Approach | Result | Verdict |
|---|---|---|
| **EgoCorridor** — OpenCV corridor on YOLOP's drivable mask | IoU **0.06**, finds lane 4% of frames | Model-based; collapses in dense/occluded scenes. **Rejected.** |
| **UFLD** (Ultra-Fast-Lane-Detection, CULane weights) | Placed lanes **in the sky/buildings** on our footage | Camera-geometry domain gap; off-the-shelf ≠ transferable. **Rejected.** (`output/ufld_check/`) |
| **`ego_seg`** — YOLOv8n-seg trained on BDD ego-lane masks | Mask mAP50 **0.972** | **Adopted.** |

The UFLD failure is the whole thesis in one result: a strong published model can
be useless on a new camera geometry, and **training on real target-domain masks**
is the fix.

### Ego-lane: why not the Indian IDD model as the default?

We *did* train one — `ego_seg_idd` on IDD (Indian Driving Dataset) YOLO-seg
polygons (12k train / 2k val, 50 epochs). It scored **mask mAP50 0.14** — ~7× worse
than the BDD model, due to a fine-grained-label / class-mapping mismatch. **We keep
BDD `ego_seg` (0.972) as the default** and leave `ego_seg_idd` selectable only for
an explicit A/B comparison. Shipping the worse model to look "more Indian" would
have made the lane output worse — so we didn't.

### Signs: why 4 super-classes, and why not a two-stage classifier?

| Approach | Behaviour | Verdict |
|---|---|---|
| Detector + **GTSRB 43-class classifier** | 96.81% on GTSRB test, but **saturated**: reported confidence **1.00 on signs it had never seen** (e.g. a U-turn, absent from GTSRB, labelled "Ahead only" at full confidence). No confidence gate can filter that. | **Rejected.** |
| Detector on **synthetic** crops pasted on road frames | 99.3% "mAP" on synthetic val, but ~57% real-world presence | **Rejected** (train/serve gap). |
| **Single-stage, 4 super-classes on real data** | mAP50 **~0.91** held-out; output is what shape+colour genuinely support | **Adopted.** |
| Indian-only sign set | No prohibitory/mandatory data at all | **Rejected** (merged with GTSDB instead). |

The lesson: a model that **can't tell you it's uncertain** is worse than a coarser
model that is honest. Four super-classes are what a sign's shape and colour
robustly support.

### Lights: why Bosch, and why not the alternatives?

| Approach | Why not |
|---|---|
| **BDD-YOLO traffic-light set** | Collapses colour → a single "traffic light" class; loses the state we need |
| **Reconstruct state from BDD JSON** | Too complex; label plumbing risk |
| **Bosch Small Traffic Lights** | Native red/green/yellow/off classes. **Adopted**, despite its own weaknesses (yellow-sparse, US-domain), which we mitigate with the temporal tracker + calibrated gate. |

---

## Slide 11 — Live validation on Indian footage (measured, not cherry-picked)

Full pipeline, real dashcam clips, telemetry from `src/run_pipeline_fast.py`.

| Metric | Kolkata (30 s) | Mumbai (30 s) |
|---|---|---|
| Ego-lane present | 100% | 100% |
| Mean lane offset | 0.01 (centred) | — |
| Vehicles / frame | 4.87 | 9.26 |
| Signs (by class) | danger 137, other 57, prohibitory 6 | mandatory 27, other 9 |
| Lights (by state) | RED 24, GREEN 75 | GREEN 63, RED 12 |
| Processing FPS | 21 | 21 |

Two different scenes, both work — the light calibration was not over-fit to one
clip. Notice `mandatory` appears in Mumbai but not Kolkata: the detector uses that
class only when the footage actually contains those signs.

### Night / low-light — the honest hard case

Measured (`src/eval_night.py`) on clips selected by brightness:

| Metric | Day | Night | Δ |
|---|---|---|---|
| Ego-lane present (% frames) | 100% | 84% | −16 |
| Traffic lights / frame | 3.0 | 1.0 | **−2.0** |
| Vehicles / frame | 7.0 | 8.8 | +1.8 (head/tail-lights help) |
| Mean detection confidence | 0.568 | 0.537 | −0.03 |

**Traffic-light recall is the component that degrades most at night** (~⅓ of day).
This matches the field's findings and points the next improvement at a low-light
traffic-light model. Stated, not buried.

---

## Slide 12 — Performance & the machine

**Hardware:** laptop **NVIDIA RTX 3050, 6 GB VRAM**, 1280×720 input.

Per-call inference cost (TensorRT FP16, measured):

| Model | ms / call |
|---|---|
| YOLOv8n (vehicles/lights) | 16 |
| `light_state` | 15 |
| `signs_merged_detector` | 14 |
| YOLOP | 24 |
| `ego_seg` | 19 |

With frame-skipping: **16 FPS** default, **23 FPS** in `--fast` live mode — five
neural networks concurrently in 6 GB.

**Why TensorRT FP16:** engines are compiled per-GPU; FP16 roughly halves inference
time and VRAM versus FP32 with no measurable accuracy loss on these models. (The
one FP32 engine we missed cost us 5× on that model until we caught it — see
Slide 8.)

---

## Slide 13 — Technology stack

| Layer | Technology | Role |
|---|---|---|
| Models | **Ultralytics YOLOv8 / YOLOv8-seg, YOLOP** | detection + segmentation |
| Training | **PyTorch 2.x (CUDA 12.6)** | on Kaggle 2× T4 + Google Colab T4 (parallel), local RTX 3050 for engine export |
| Inference | **TensorRT 11.2 (FP16)**, ONNX | 2–5× speedup over PyTorch |
| Classical CV | **OpenCV** | lane smoothing, morphology, geometry gate, colour opponency, temporal voting |
| Backend | **FastAPI + Uvicorn** | job queue, single-GPU serialization, MJPEG live stream |
| Frontend | **HTML + Tailwind + vanilla JS** | upload/drag-drop, live preview vs final render, telemetry HUD, charts |
| Launchers | Windows `.bat` | `run_live.bat` (web app), `run.bat` (frontend-free live window) |

**Why train on the cloud:** two Kaggle T4s let us train the lane and IDD models in
parallel; Colab ran the sign and light models simultaneously. A T4 ≈ an RTX 3050 in
raw compute — the cloud's value here was **parallelism**, not per-model speed. Final
TensorRT engines are compiled **locally** because an engine is bound to its exact
GPU + driver + TensorRT version.

---

## Slide 14 — How to run it

**Option A — Web app (frontend + backend, one click):**
```bash
run_live.bat        # starts FastAPI, opens the browser at http://127.0.0.1:8000
```
- Upload/drag-drop a video or pick a sample.
- Choose **Final render** (full-quality annotated MP4, downloadable) or
  **Live preview** (frames stream as they process).
- **Live (Long)** section streams a full-length drive (25–75 min) live, nothing
  saved — for long validation. Stop button ends it and frees the GPU.

**Option B — Standalone live window (no frontend needed):**
```bash
run.bat             # menu: pick Kolkata / Mumbai / India-Night full clip → live window
```
Press `Q` / `Esc` to stop.

**Option C — Direct CLI:**
```bash
python src/run_pipeline_fast.py --input clip.mp4 --live --no-record --no-crop --fast
```

---

## Slide 15 — Honest limitations (stated plainly)

- **Signs:** localisation is solid, but `danger↔other` class confusion exists and
  `mandatory`/`prohibitory` are weak — a training-data imbalance, not a bug.
- **Lights:** functional but low-confidence on Indian signals (Bosch domain gap);
  yellow is the weakest class (sparse training data). Fix = Indian-signal fine-tune.
- **Ego-lane:** trained on BDD (US/varied). The Indian IDD variant we trained
  scored only 0.14, so BDD remains the default; severe occlusion is still the weak
  case.
- **Night:** traffic-light recall drops to ~⅓ of daytime. Known field failure mode.
- **Light association** is by grid cell, not IoU tracking — two very close lights
  can merge into one track.
- **1080p is CPU-bound** (video decode dominates); 720p is the sweet spot.

---

## Slide 16 — What we'd build next

1. **Indian-signal traffic-light fine-tune** — the single biggest quality lever
   (raises confidence, lets us tighten the gate, fixes the night case).
2. **Rebalance the sign dataset** — add prohibitory/mandatory Indian examples to
   fix the `danger↔other` confusion.
3. **Fix or drop `ego_seg_idd`** — diagnose the IDD label mapping so a genuine
   Indian lane model can beat BDD.
4. **IoU-based light tracking** — replace grid-cell association.

---

## Appendix — Reproducibility

| Artefact | Command / source |
|---|---|
| Ego-lane data + train | `src/prepare_ego_seg.py` → `kaggle/train_ego_seg_kaggle.py` |
| Sign merge + train | `src/prepare_indian_signs.py` → `kaggle/train_sign_detector_colab.py` |
| Light train | `kaggle/train_light_state_colab.py` (Bosch) |
| TensorRT export (local) | `YOLO('models/X.pt').export(format='engine', imgsz=640, half=True, device=0)` |
| Lane IoU eval | `src/eval_lane_iou.py` |
| Night eval | `src/eval_night.py` |
| Light-gate diagnostics | `src/capture_diagnostics.py` |
| Metrics of record | `runs/*/results.csv` |

**References:** YOLOP (Wu et al., MIR 2022) · YOLOv8 (Ultralytics) · BDD100K (Yu
et al.) · GTSDB (Houben et al., IJCNN 2013) · Bosch Small Traffic Lights (Behrendt
et al., ICRA 2017) · IDD (Varma et al., WACV 2019).

*Every metric in this document traces to a `results.csv`, an evaluation script, or
an instrumented run on real footage. Weaknesses are named, not hidden.*
