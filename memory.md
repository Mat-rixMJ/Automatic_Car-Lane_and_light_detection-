# Memory — Global Python Packages Reference

Packages that were installed globally before cleanup (2026-08-16).
Reinstall what you need later from this list.

## Commonly Used (Kept or Reinstall First)

```bash
pip install numpy pandas matplotlib scipy seaborn pillow requests pyyaml
pip install ipython jupyter jupyterlab notebook
pip install python-dotenv
```

## ML/AI (Install in venv per-project)

```bash
# PyTorch (CUDA 13.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# TensorFlow
pip install tensorflow

# Ultralytics (YOLOv8)
pip install ultralytics

# HuggingFace
pip install transformers tokenizers safetensors huggingface_hub

# ONNX
pip install onnxruntime-gpu

# OpenCV
pip install opencv-python
```

## Finance/Trading

```bash
pip install yfinance pandas-ta ccxt backtrader polars dhanhq fyers_apiv3
```

## Web/API

```bash
pip install fastapi uvicorn starlette flask streamlit
pip install httpx aiohttp beautifulsoup4 playwright
pip install pydantic sqlmodel sqlalchemy
pip install anthropic openai-whisper google-generativeai
```

## Data/DB

```bash
pip install openpyxl xlsxwriter sqlite-utils psycopg-binary mysql-connector-python
pip install pyarrow
```

## Media/Download

```bash
pip install yt-dlp gallery_dl ffmpeg-python mutagen
pip install openai-whisper tiktoken
```

## Dev Tools

```bash
pip install pytest hypothesis gitpython loguru rich typer click
pip install boto3 playwright praw python-telegram-bot
```

## Misc

```bash
pip install sympy numba networkx nltk faker holidays reportlab fpdf
```

---

# Pipeline Performance & Accuracy Findings (2026-08-17)

Measured on RTX 3050 6GB Laptop, 1280x720 source. Don't re-guess these — they're
profiled numbers from `src/profile_pipeline.py` and `src/diagnose_detection.py`.

## FPS: where the time actually goes

Per-frame cost, every stage on every frame, before optimisation:

| stage | cost | note |
|---|---|---|
| YOLOP (PyTorch) | 67 ms | 58% of total — the real bottleneck |
| sign detector | 15 ms | |
| yolov8 | 13 ms | |
| imshow | 7 ms | |
| draw | 6.5 ms | |
| write | 5 ms | recording to file |
| read | 1.6 ms | |

After exporting YOLOP to TensorRT, its inference dropped 67 ms -> 23 ms. That one
change was worth more than every other optimisation combined.

**Lesson: the first optimisation attempt targeted the 6.5 ms draw code while the
67 ms model sat untouched. Profile before optimising.**

Final: **16 FPS -> 28 FPS** live at 720p (target was 25+).

Skip intervals are set from measured cost, not guessed: `YOLOP/4`, `Detection/3`.

## Traffic lights were missed because of downscaling, not model weakness

Traffic light box height in native 720p pixels: median 34, p25 30, only 3% under 25px.
After the 512p downscale + 480 letterbox, **the median light became 12.9px** — too
small for YOLOv8n to fire on.

Detection strategy comparison over 400 frames:

| strategy | detections | frames with TL |
|---|---|---|
| 512p downscale @ 480 | 89 | 22.0% |
| **native 720p @ 640** | **308** | **53.5%** |
| upper-centre crop @ 640 | 113 | 22.2% |

Fix: run vehicle/light detection on the **native frame at imgsz=640**. The TRT
engine input size is fixed at build time, so the engine must be rebuilt to match
(`src/export_tensorrt.py`).

Progression: 9.4% -> 23.6% -> **38.3%** frame presence (384 -> 2444 detections).

Remaining ceiling: YOLOv8n-COCO is genuinely weak on distant lights. A dedicated
traffic-light model is the next step if 38% isn't enough.

## The sign classifier CANNOT be trusted for exact names

GTSRB has 43 classes and **no U-turn sign**. Any sign outside those 43 is
out-of-distribution and still gets assigned one of the 43.

Measured over 314 real detected crops:
- median confidence: **1.00**
- median top1-vs-top2 margin: **1.00**
- most-assigned label: "Ahead only" (102x)

The model is saturated, so **a confidence or margin gate cannot reject OOD input**.
A real U-turn sign came back as "Ahead only" with full confidence.

Therefore the pipeline displays the **detector super-class** (prohibitory /
mandatory / danger) as the primary label — that is always correct — and appends the
GTSRB name only when it agrees with the super-class, suffixed with `?` to mark it
unverified.

Proper fix (not done): retrain the detector on real per-class annotated German
sign photos so exact names come from the detector, not a separate classifier.

## TensorRT gotchas

- TRT 11 removed `BuilderFlag.FP16` and `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`.
  Use `builder.create_network()` with no flags; TF32 is on by default.
- Ultralytics' own `.export(format="engine")` requires `nvidia-modelopt`, which
  failed to install. Workaround: export to ONNX via ultralytics, then build the
  engine directly with the TensorRT Python API (`src/export_tensorrt.py`).
- **TRT engines drop class-name metadata** — `model.names` returns `class2`, `class7`.
  Names must be hardcoded, otherwise overlays display "class2" instead of "car".
- Engine input resolution is baked in at build time. Changing `imgsz` at inference
  requires rebuilding the engine.

## Lane detection: OpenCV's actual role

Classical OpenCV (Canny + Hough) is **worse** than YOLOP at finding lanes — brittle
to shadows, faded paint, curves, occlusion. Do not replace YOLOP with it.

OpenCV *is* the right tool for refining YOLOP's output: warp the mask to bird's-eye,
fit a 2nd-order polynomial per side, EMA across frames. This turns the patchy
flickering mask into smooth stable curves and yields the ego lane offset that
lane-departure warning needs. See `src/lane_fit.py`, checked by `src/test_lane_fit.py`.

## Validation

`src/validate_pipeline.py` runs detection on every frame (no skipping) and prints
per-class rates with pass/fail gates. Current state over 3540 frames:

```
road 92.6% | lane 91.9% | vehicle 92.8% | traffic_light 38.3% | sign 57.0%
cars 10.35/frame | TL state classified on 85.6%
all 4 gates PASS
```

Note: the sign detector's 99.3% mAP is against the **synthetic** training set
(GTSRB crops composited onto BDDA road frames), not real German road photos. The
real-world figure is the 57% frame presence above.

---

# Lane Fitting: first design was wrong, and the tests hid it (2026-08-17)

## What the user saw
"just 2 yellow lines which move in any direction" — the fitted lanes drifted
independently of the road.

## Why the unit tests said 8/8 PASS anyway
The tests generated **synthetic lane masks using the same perspective assumption
the fitter used**. They confirmed my assumptions instead of testing them. A test
that shares the code's assumptions proves nothing.

**Rule: validate components against real model output, not self-generated data
that matches your own design.**

## The four real bugs (found via src/diagnose_lanes.py on real YOLOP masks)

```
 frame  lane px  ymin%  ymax%  warped px   L px   R px  fit?
   200       38     70     81        104    104      0  True
   900     6558     65     96       8071   7721    350  True
  1800     4702     67     87      23037  11782  11255  True
```

1. **Stale fits were drawn.** Frame 200 had 0 right-side pixels but reported
   `fit=True` — drawing a curve from an older frame. `missed` was reset whenever
   *total* pixels cleared the threshold, so per-side starvation never expired.
   This was the "lines moving in any direction".
2. **Midpoint L/R split is invalid.** YOLOP marks EVERY lane line in view, not just
   the ego lane's two. Counts like 7721 vs 350 show one side absorbing several
   distinct lines, so one polynomial was fitted through pixels from multiple lanes.
3. **Massive extrapolation.** Lane pixels occupy only y = 65-99% of frame height,
   but the curve was evaluated over y = 0-719. Most of every drawn line was
   extrapolated beyond any supporting data.
4. Frames with 38 lane pixels still drew lanes.

## Replacement design (src/lane_fit.py)
Dropped the bird's-eye polyfit entirely. Now:
- `connectedComponentsWithStats` on the mask -> one component per painted line
- fit `x = f(y)` per component (2nd order if tall, else linear)
- **evaluate only across that component's own y-extent** — no extrapolation
- ego lane = nearest fitted line either side of the camera axis that reaches low
  in the frame; offset is None unless BOTH exist
- nothing is retained between frames, so a lane-free frame draws nothing
- the raw YOLOP mask is still drawn always, so lanes stay visible even if the fit
  declines to commit

## Tuning: the residual gate mattered, the kernel didn't
Sweep over 400 real masks (`src/tune_lane_fit.py`). Rejection breakdown at the
original settings: 68.3% `too_few_px`, 14.6% `too_short`, 3.8% `bad_residual`,
only 13.4% accepted.

```
    kernel  minpx  span  resid  fitted%   ego%  lines/f
    (3, 9)     45    18     12      78%    22%      1.7   original
   (5, 41)     45    18     12      78%    23%      1.6   taller kernel alone: ~nothing
   (5, 41)     45    18     20      83%    32%      2.0   relaxing residual: +10%
   (5, 61)     40    15     20      86%    34%      2.1
   (7, 81)     40    15     25      85%    34%      2.1   no further gain
```

I assumed dash-fragmentation was the limiter; it wasn't. The residual threshold
rejecting slightly-curved real lines was. Chose `(5,41)/40/15/20` — skipped the
61-81px kernels since +2% wasn't worth the risk of bridging separate lines.

Result on real footage: **fitted 91%, ego pair 50%** (was 89% / 42%).

Ego-pair 50% is honest for dense city driving — both boundaries are frequently
occluded by traffic, or absent at intersections, crosswalks and tram crossings.

## Test suite now (src/test_lane_fit.py, 10/10)
Each check targets a bug that actually occurred, plus one regression test that
runs real YOLOP masks through the fitter and asserts:
- no fit when pixels < MIN_PX
- no offset without a genuine ego pair
- every drawn point inside its component's real extent
Two test failures during this work were the *test's* fault (brush radius, then the
close kernel legitimately extending the component) — worth checking which side is
wrong before changing code.

---

# Ego corridor + signal filtering, measured (2026-08-17)

## Method that actually worked
I could not eyeball frames, so every visual complaint was converted into a metric
(`src/capture_diagnostics.py`): green **spill** off the road surface, corridor
**width** as a fraction of frame, **width variation**, centre **jitter** px/frame,
and for signals **below-horizon** count and **state flip** rate. Three tuning
iterations were then driven by those numbers, not by looking.

## Key discovery: lane-line pairing is not viable on this footage
`src/measure_lane_width.py` — across ~2800 frames of Frankfurt + 3 BDDA clips,
only **4 frames** produced two fitted lane lines that both span a common height
range. Earlier "57% / 72% ego pair" figures were counting pairs whose two lines
existed at *different* heights, so they never actually bracketed the vehicle.

YOLOP marks lane pixels fine, but the ego lane's two boundaries are rarely both
visible and unbroken at once (dashes, occlusion, single-sided markings,
intersections). **Conclusion: build the corridor from the drivable-area mask
(present >92% of frames) and use lane markings only to refine it.**

Those 4 genuine pairs measured ~44% of frame width at y=95%h — so a wide corridor
low in the frame is NORMAL for a dashcam. My earlier "too wide >45%" metric was
itself wrong.

## The DA-fallback trap
First attempt at a fallback drove ego-pair 0% -> 99% and I reported that as a win.
It was not: the corridor was 47-66% of frame width, i.e. it had gone back to
outlining the whole road — the exact bug being fixed. **Coverage went up while
correctness went down. Always pair a coverage metric with a correctness metric.**

## Occlusion was the cause of collapsed corridors
Taking the *longest* contiguous road run per scanline fails in traffic: vehicles
punch holes in the DA mask, so the longest run is often a fragment beside a car.
Result was a 7%-of-frame sliver on BDDA1003. Fixes:
  * horizontal MORPH_CLOSE (121x9) to bridge vehicle-shaped holes
  * pick the run **containing the tracked anchor**, not the longest
  * carry the bottom anchor across frames (re-anchoring to frame centre each
    frame caused snapping to run edges — 52px/frame jitter)

A hard width floor (`MIN_W_FRAC 0.55`) forced the corridor wider than the road and
pushed spill to 21-31%. Lowered to 0.30 once bridging removed the sliver cause.

## Results across three iterations

```
                 Frankfurt              BDDA100              BDDA1003
              orig   v2   v3   v4    orig   v2   v3   v4   orig   v2   v3   v4
width          47%  20%  33%  27%     66%  38%  43%  42%    48%   7%  24%  13%
width var      43%  82%  32%  58%     23%  27%  15%  18%    46%  26%   2%   2%
jitter (px)     34   21   12   9.8     62   10  3.3  3.0     60   30   52   30
spill          11%  13%  21%  13%      8%   6%   7%   6%    14%  18%  31%  23%
```

Signals (`src/traffic_light.py`) — geometry gate + temporal state voting:
```
                  before   after
below horizon      53%      0%     <- was the dominant false positive
state flips        11%      1%
kept after gates    —      44-60% of raw candidates
```

FPS held at **28** at 720p.

## Honest remaining weakness
BDDA1003 (dense urban, heavy occlusion) is still the worst clip: 13% corridor
width, 23% spill, 30px jitter. Its DA mask is small and fragmented, so the
corridor has little to lock onto. Frankfurt and BDDA100 are good.

The real ceiling: this is a *model-based corridor refined by segmentation*, not a
true ego-lane detection. A model trained to output ego-lane boundaries directly
(CLRNet, UFLD, or YOLOP fine-tuned on ego-lane labels) is the upgrade path.

---

# Session close — state as of 2026-08-17

## Where the project stands

Scope narrowed to **ego-lane + traffic-light detection** (plus vehicles). Sign
recognition removed from the pipeline. Running at **28-32 FPS at 720p** on the
RTX 3050 6GB, all six validation gates passing.

Pushed to GitHub: `Mat-rixMJ/Automatic_Car-Lane_and_light_detection-`
(note the repo name genuinely ends in a hyphen — omitting it gives
"Repository not found"). Two commits on `main`, 0.27 MB, source + config + docs
only.

## Active pipeline

```
YOLOP (TRT, 384)      -> drivable area + lane pixels   25 ms, every 4th frame
YOLOv8n (TRT, 640)    -> vehicles + traffic lights     25 ms, every 3rd frame
EgoCorridor (OpenCV)  -> ego lane + LDW offset         part of the 10.7 ms
TrafficLightTracker   -> gated, temporally-voted R/Y/G  "
```

Entry point: `src/run_pipeline_fast.py`, or `run.bat` (edit the VIDEO path).

## What is NOT in git, and why

| Excluded | Size | Reason |
|---|---|---|
| `CARLA_0.9.15/` | 18 GB | Simulator install, downloadable |
| `.venv/` | 7.6 GB | Rebuildable from requirements.txt |
| `BDDA/`, `data/` | 6.6 GB | Datasets, downloadable |
| `output/`, `runs/` | 3 GB | Generated |
| `downloads/` | 2.3 GB | Test videos |
| `models/` | 382 MB | See below |

**TensorRT engines must never be committed.** They are compiled for a specific GPU
architecture + TensorRT version + driver, so an engine built here fails elsewhere.
Also `yolov8n.engine` is 166 MB, over GitHub's 100 MB per-file hard limit.

Verified that a fresh clone works: with `models/` empty, Ultralytics auto-downloads
`yolov8n.pt` to the absolute path it is given, and YOLOP comes via `torch.hub`
(which clones the repo and loads `weights/End-to-end.pth`, 91.3 MB, cached at
`~/.cache/torch/hub/hustvl_YOLOP_main/`).

GitHub throttles direct raw download of that 91 MB YOLOP weight file — it returns
503. The README points at `torch.hub` instead of a raw link. Verified working URL
for YOLOv8n: `github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt`
(200, 6.2 MB).

## Two irreplaceable artifacts sitting in models/, NOT committed

These cannot be re-downloaded — only retrained:
- `sign_classifier.pth` (2.9 MB) — GTSRB CNN, 96.81% official test set
- `german_sign_detector.pt` (5.9 MB) — YOLOv8n, 99.3% mAP on synthetic data

Combined 8.8 MB, well within GitHub limits. Not needed to run the pipeline since
sign detection is out of scope. **Decision deferred** — either whitelist these two
in `.gitignore` or attach to a GitHub Release. Worth doing before the machine is
ever wiped.

## Next goal

Phase 7 in `goal.md`, full plan in `future.md`: train a **custom traffic-light
detector** (YOLOv8n fine-tune, 4 classes red/yellow/green/off) on LISA + Bosch.
This puts a self-trained model in the active pipeline and fixes two documented
limitations at once — small-light recall, and the colour-opponency heuristic.

Baseline to beat is already measured, so the comparison is ready to run.

Open question flagged in `future.md`: **does the assignment mandate
TensorFlow/Keras?** This stack is PyTorch throughout. If Keras is required, the
small traffic-light state classifier is the one component trivial to port.

## Loose ends

- `git config user.name` / `user.email` were never set locally, so the two commits
  used whatever global values exist. Worth checking attribution.
- `src/` contains a lot of exploratory and superseded scripts (`pipeline.py`,
  `run_pipeline.py`, `analyze_output.py`, `validate_all.py`, several one-off
  diagnostics, plus CARLA and depth-estimation code that is unused). All committed.
  Cleaning these up would make the repo easier to read for a marker.
- Two stray weight files sit in `src/` from training runs (`yolov8n.pt`,
  `yolo26n.pt`, ~11 MB) — gitignored, but they belong in `models/`.
- CARLA integration (`carla_bridge.py`, `carla_demo.py`, `carla_controller.py`) was
  built and partially working, then set aside because autopilot quality was poor
  for demos. Real dashcam footage is used instead.

## Method notes worth carrying forward

The single most valuable habit this session: **convert every complaint into a
number before touching code.** Nearly every assumption I made without measuring
turned out wrong —

- assumed the drawing code was the bottleneck; it was 6.5 ms against YOLOP's 67 ms
- assumed dashed-line fragmentation limited lane fitting; the residual gate did
- assumed a taller morphological kernel would bridge dashes; it changed nothing
- assumed lane-line pairing was tunable; it works in 4 frames out of 2800
- assumed higher ego-pair coverage meant better output; it meant worse

Each of those was caught by measuring, and would have been shipped as a "fix"
otherwise.
