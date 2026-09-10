# Train CarLaneI models on Kaggle (free 2×T4 GPU)

Faster than the laptop RTX 3050 and keeps your machine free. Each notebook
**self-downloads its dataset** via the Kaggle API (Internet ON), trains, and writes
a `.pt` to `/kaggle/working/` for download.

## The four training jobs (run one session at a time)

| # | Notebook | Dataset (self-downloaded) | Output `.pt` |
| - | -------- | ------------------------- | ------------ |
| 1 | `train_ego_seg_kaggle.ipynb` | BDD drivable masks (US roads) | `ego_seg.pt` |
| 2 | `train_sign_detector_kaggle.ipynb` | GTSDB (German) + Indian dashcam signs | `sign_detector.pt` |
| 3 | `train_light_state_kaggle.ipynb` | Bosch Small Traffic Lights (YOLO) | `light_state.pt` |
| 4 | `train_ego_seg_idd_kaggle.ipynb` | IDD YOLO-Seg (Indian roads) | `ego_seg_idd.pt` |

Kaggle free tier is ~30 GPU-hours/week; each job fits in one ~7-8h session. Run them
sequentially (one GPU kernel at a time).

### What each model is for
- **1 — ego_seg (BDD):** drivable-area corridor, US-road baseline. IoU 0.59 vs 0.06 heuristic.
- **2 — sign_detector:** one detector over 4 shape+colour super-classes
  (prohibitory / mandatory / danger / other), trained on **both** German (GTSDB) and
  Indian dashcam photos so it works across both countries honestly.
- **3 — light_state:** emits the light **colour state** directly (red/green/yellow/off).
  Replaces the colour heuristic that the night eval flagged as weakest.
- **4 — ego_seg_idd:** same drivable task as #1 but trained on the India Driving
  Dataset — the domain match for the Indian showcase footage (Kolkata/Mumbai/NH-44).

## Files here
- `train_*_kaggle.ipynb` — the notebooks to run on Kaggle.
- `train_*_kaggle.py` — the same code as plain scripts (source of truth).
- `make_notebook.py` — regenerates a notebook from its `.py`
  (`python make_notebook.py ego|sign|light|idd`, or no arg for all).
- `kernel-metadata*.json` — per-notebook kernel config for the "Run with Kaggle" ext.
- `selfcheck_sign_merge.py`, `selfcheck_new_notebooks.py` — local (no-GPU) checks of
  the risky mapping logic. Run before spending a session.

## EASIEST: the "Run with Kaggle" VS Code/Kiro extension (installed)

Pushes the notebook to a remote Kaggle GPU kernel and streams logs back.

1. **Settings** (Ctrl+, → "runWithKaggle"):
   - `runWithKaggle.accelerator` → **`gpu-t4`**
   - `runWithKaggle.enableInternet` → **true** (notebooks self-download data + weights)
   - `runWithKaggle.username` → **matrixmj**
2. Command Palette → **"Validate Kaggle Credentials"** (token at `~/.kaggle/kaggle.json`).
3. Open the notebook you want → Command Palette → **"Run Notebook with Kaggle"**.
4. Watch with **"Show Live Kaggle Logs"**.
5. Download the `.pt` from the run's output (the extension shows the kernel URL).

> No stop command in the extension — to stop a run, delete the notebook/kernel on
> kaggle.com.

## ALTERNATIVE: browser (account: matrixmj)
1. kaggle.com → Create → New Notebook → **File → Upload Notebook** → pick the `.ipynb`.
2. Settings → Accelerator → **GPU T4 x2**, Internet **ON**.
3. **Run All**. First cell installs ultralytics; the data build takes a few minutes.
4. **Output** tab → download the `.pt`.

## First-run things to eyeball (per notebook)
- **sign:** prints the **Indian class → superclass** table. If a sign clearly landed in
  the wrong bucket, tell me and I add a one-line override (`INDIAN_OVERRIDES`).
- **light:** prints boxes-per-class (red/green/yellow/off). Hue jitter is off on purpose
  (colour is the class).
- **idd:** prints the **IDD class list + chosen drivable class**. If the keyword picked
  the wrong class, set `DRIVABLE_OVERRIDE` at the top of the script and regen the notebook.

## Back on your machine (after a `.pt` lands)
Drop the downloaded weight into `models/`, then tell me — I'll:
1. Export the TensorRT engine, e.g.
   `YOLO('models/ego_seg.pt').export(format='engine', imgsz=640, half=True, device=0)`
   (segmentation) or the detect equivalent for signs/lights.
2. Wire it into the pipeline and re-run the relevant eval
   (`eval_lane_iou.py`, `eval_night.py`, or a val pass) for the honest numbers.

## If a notebook can't find its data
Each notebook auto-detects images/labels under `/kaggle/input` (and falls back to a
Kaggle API download). On failure it prints every folder that has images/labels so you
can see what got attached.

## Config (edit the top of each `.py`, then `python make_notebook.py <job>`)
- ego / idd: imgsz 640, batch 32 (2×T4), ~40-50 epochs, road-safe aug (no vertical flip).
- sign: imgsz 640, epochs 80, **no horizontal flip** (a mirrored arrow is a wrong sign).
- light: imgsz 640, epochs 60, **no hue jitter** (colour is the class), h-flip ok.
