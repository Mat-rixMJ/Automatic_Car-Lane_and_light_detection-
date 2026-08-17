# Future Work — Training a Custom Model

Written 2026-08-17. Picking this up in 2-3 days.

---

## Why this is needed

The assignment requires a **custom trained model**. The current pipeline has none:

| Component | Model | Trained by us? |
|---|---|---|
| Lane + drivable area | YOLOP (BDD100K pretrained) | No |
| Vehicles + traffic lights | YOLOv8n (COCO pretrained) | No |
| Ego-lane corridor | `ego_corridor.py` — OpenCV geometry, no model | N/A |
| Traffic-light state | `traffic_light.py` — colour opponency heuristic | N/A |

Two models *were* trained during development but are **not in the active path**:
- `sign_classifier.pth` — GTSRB CNN, 96.81% on the official test set
- `german_sign_detector.pt` — YOLOv8n, 99.3% mAP on synthetic data

Sign recognition was removed from the pipeline (see README for the reason), so
neither counts as a deliverable for the current scope.

## Two assets that make this cheap

1. **A working measurement harness.** `capture_diagnostics.py`,
   `validate_pipeline.py`, `profile_current.py` already produce baseline numbers.
   A custom model can be *proven* better with the same metrics, on the same clips.
2. **A proven training workflow.** `train_combined_detector.py` already took a
   dataset to a trained YOLOv8n and into TensorRT. The path is known to work on
   this hardware.

---

## Options considered

| Option | What to train | Replaces | Effort | Risk |
|---|---|---|---|---|
| **A. Traffic-light state CNN** | Small classifier: red/yellow/green/off | The colour heuristic in `traffic_light.py` | ~1 day | Low |
| **B. Traffic-light detector** | YOLOv8n fine-tune, detect + state in one pass | YOLOv8n-COCO class 9 | 1-2 days | Low |
| **C. Ego-lane segmentation** | Small U-Net or YOLOP fine-tune outputting the *ego lane* | The geometry in `ego_corridor.py` | 3-5 days | Medium |
| **D. Lane net from scratch** | U-Net on TuSimple | YOLOP | 4-6 days | High |

### Decision: B first, C as stretch

**B is chosen** because it fixes two limitations already documented in the README
and lands a custom model in the **active pipeline**, not a side branch:

1. *"Traffic-light recall is capped by YOLOv8n-COCO on small distant lights"* — a
   dedicated detector trained on light-specific data raises recall.
2. *"Light state uses colour opponency, so it can struggle at night and with
   backlighting"* — training state as detector classes removes the heuristic.

**D is rejected as the headline.** Training lane segmentation from scratch
reinvents YOLOP, worse. Still worth doing as an *ablation* ("we trained our own
and compared against YOLOP") but not as the main contribution.

---

## Datasets

| Dataset | Content | Access | Fit |
|---|---|---|---|
| **LISA Traffic Light** | 43k frames, 113k annotated lights, day + night, San Diego | Kaggle, easy | Start here — fastest to obtain |
| **Bosch Small Traffic Lights** | 13.4k images, 24k lights, includes lights down to ~8px | Registration | Best for the *small distant light* weakness specifically |
| **DTLD (DriveU Traffic Light)** | ~40k images, 232k lights, **German** | Registration, research use | Best match for the Frankfurt footage |
| **TuSimple** | 6.4k images, ego + adjacent lanes labelled | Open | For option C |
| **CULane** | 133k images, urban, harder scenarios | Open | For option C |

Plan: start with **LISA**, then add **Bosch** because small-light recall is the
documented weak point. Get **DTLD** if registration comes through, since it is
German and matches the primary test footage.

### Legitimate shortcut: bootstrap in-domain labels

Run the current YOLOv8 over the Frankfurt and BDDA clips, crop every detected
light box, then hand-verify the red/yellow/green label. This gives cheap
**in-domain** training data.

Standard pseudo-labelling, with one rule: **do not evaluate on it.** Those labels
inherit the baseline's biases, so measuring against them would just confirm the
baseline. Held-out evaluation must use real dataset labels.

---

## Feasibility on the RTX 3050 6GB

Grounded in a measured reference point: the sign detector trained 2000 images x
60 epochs in ~15 min at 480px, batch 8, using 0.62 GB VRAM.

| Task | Config | Expected time |
|---|---|---|
| YOLOv8n fine-tune (B) | 640px, batch 8, ~10k images, 60 epochs | 2-4 hrs |
| State classifier CNN (A) | 64x64, batch 64 | ~10 min |
| Lane segmentation (C) | 288x800, batch 4 | 6-10 hrs |

All fit within 6 GB. Note from experience: batch 16 at 640px **ran out of memory**
during the sign-detector work — batch 8 is the safe ceiling at that resolution.

---

## Baseline to beat

Already measured, so the comparison is ready. From `capture_diagnostics.py` and
`validate_pipeline.py`:

| Metric | Current (YOLOv8n-COCO + heuristic) |
|---|---|
| Traffic-light frame presence | 38.3% |
| Light state classified | 85-98% depending on clip |
| State flips between frames | 1% (after temporal voting) |
| Detections below horizon | 0% (after geometry gate) |
| Vehicles | 8.8-13.3 per frame |
| FPS at 720p | 28-32 |

Per-stage timing to compare against, from `profile_current.py`:

| Stage | Cost/frame |
|---|---|
| YOLOP (TensorRT) | 25.0 ms |
| YOLOv8n (TensorRT) | 24.9 ms |
| Corridor + light tracking | 10.7 ms |

---

## Sequence

```
Phase 1  Acquire LISA (+ Bosch). Convert to YOLO format.
         Split train/val/TEST — test set is held out and never tuned on.
Phase 2  Fine-tune YOLOv8n on 4 classes: red / yellow / green / off
Phase 3  Evaluate on the held-out test set:
           mAP50, mAP50-95, per-class recall,
           and recall broken out by box size (the small-light case)
Phase 4  Swap into the pipeline, export TensorRT, re-run capture_diagnostics.py
         on the SAME clips as the baseline
Phase 5  Write up: baseline vs custom, identical clips, identical metrics,
         including the FPS cost
Stretch  Option C — ego-lane model on TuSimple, removing the geometric corridor
```

## Evaluation discipline

Notes to self, learned the hard way in this project:

1. **Held-out test set, never tuned on.** The GTSRB work showed the gap plainly:
   99.15% on the train split vs 96.81% on the official test set.
2. **Report the FPS cost.** A more accurate model that drops the pipeline to
   15 FPS is a real trade-off. Stating it is a strength, not a weakness.
3. **Pair every coverage metric with a correctness metric.** An earlier change
   drove "ego pair found" from 0% to 99% while making the output *worse* — it had
   gone back to outlining the whole road. Coverage alone is a trap.
4. **Be explicit about synthetic-vs-real gaps.** The sign detector scored 99.3%
   mAP on synthetic composites but only ~57% frame presence on real footage. That
   gap is worth writing up — it demonstrates understanding of evaluation, and
   hiding it would be dishonest.
5. **Don't test a component against its own assumptions.** The first lane-fitter
   test suite generated synthetic masks from the same perspective model the fitter
   used. It passed 8/8 while the fitter was visibly broken on real video.

---

## Open question before starting

**Check whether the assignment mandates TensorFlow/Keras.** The original brief
mentioned it; this stack is PyTorch throughout.

If Keras is a hard requirement, **option A (the small state classifier) is the one
piece that is trivial to write in Keras** — it is a ~64x64 input, 3-4 class CNN.
The detector and lane work are much more natural in PyTorch and would be painful
to port.

---

## Files that already exist and will be reused

| File | Reuse for |
|---|---|
| `src/train_combined_detector.py` | Template for the YOLOv8 fine-tune — dataset prep, YAML, training args |
| `src/sign_classifier.py` | Template for a small CNN classifier (option A) |
| `src/export_tensorrt.py` | Converting the custom model to a TensorRT engine |
| `src/capture_diagnostics.py` | Before/after comparison on identical clips |
| `src/validate_pipeline.py` | Per-class rates with pass-fail gates |
| `src/profile_current.py` | Confirming the FPS cost of the swap |
| `src/traffic_light.py` | The heuristic being replaced — keep for A/B comparison |
