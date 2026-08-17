# Goal — Automatic Car Lane & Traffic Light Detection

Status as of 2026-08-17.
Scope: **ego-lane detection + traffic-light detection with state**, plus vehicles.
Traffic-sign recognition was built, evaluated, then removed — reasons in README.

---

## Phase 1: Environment & Setup — DONE

- [x] Project structure
- [x] Python 3.10 venv (`.venv`) — chosen for CARLA 0.9.15 compatibility
- [x] PyTorch + CUDA
- [x] OpenCV, Ultralytics, ONNX Runtime
- [x] TensorRT 11.2.1 installed and working
- [x] Config files
- [x] `memory.md` engineering log

---

## Phase 2: Models — DONE

- [x] YOLOP loads via `torch.hub` (lane + drivable area, BDD100K pretrained)
- [x] YOLOv8n COCO (vehicles + traffic lights, class 9 confirmed)
- [x] YOLOP exported to TensorRT — **67 ms to 25 ms**, the single biggest speed win
- [x] YOLOv8n exported to TensorRT at 640 (native frame, not downscaled)
- [x] Verified both fall back to PyTorch automatically when engines are absent

---

## Phase 3: Lane Detection — DONE

- [x] YOLOP lane + drivable-area masks working
- [x] Discovered lane-line pairing is **not viable** — only 4 of ~2800 frames gave
      two lines spanning a common height range
- [x] `EgoCorridor` built from the drivable-area mask instead (present >92% of frames)
- [x] Morphological occlusion bridging — vehicles were fragmenting the road mask
- [x] Perspective width prior, anchored and temporally smoothed
- [x] Highlights **only the ego lane**, not the whole road surface
- [x] Lane departure warning from ego offset
- [x] Centre jitter cut from 34-62 px/frame to **3-10 px** on city and highway

---

## Phase 4: Traffic Light Detection — DONE

- [x] Detection on the **native frame at 640** — a 512p downscale shrank the median
      light from 34 px to 13 px and lost 2.4x of them
- [x] Geometry gate — below-horizon false positives **53% to 0%**
- [x] Colour-opponency state classification (red / yellow / green)
- [x] Temporal state voting — frame-to-frame flipping **11% to 1%**
- [x] Ghost suppression (single-frame detections dropped)

---

## Phase 5: Performance — DONE

- [x] Profiled per stage rather than guessing (first attempt optimised 6.5 ms of
      drawing while YOLOP sat at 67 ms untouched)
- [x] Both models TensorRT-compiled
- [x] Frame skipping from measured cost: YOLOP/4, YOLOv8/3, results cached
- [x] **28-32 FPS live at 720p** on an RTX 3050 6GB laptop (target was 25+)
- [x] Established that 1080p is CPU-bound on video decode, not GPU-bound

---

## Phase 6: Validation & Delivery — DONE

- [x] `capture_diagnostics.py` — spill, corridor width, jitter, light stability
- [x] `validate_pipeline.py` — per-class rates with pass/fail gates, every frame
- [x] `profile_current.py` — per-stage timing
- [x] `test_lane_fit.py` — 10/10, including a real-YOLOP-mask regression test
- [x] Tested on Frankfurt (city), BDDA highway and BDDA urban clips
- [x] `run.bat` one-click launcher
- [x] README rewritten to match what actually exists
- [x] Git repo pushed — 0.27 MB, no datasets/weights/engines committed

---

## Phase 7: Custom Model — NEXT

**This is the current goal.** Full plan in [`future.md`](future.md).

The assignment requires a custom trained model. The active pipeline currently has
none — YOLOP and YOLOv8n are both pretrained, and the two models that *were*
trained (sign classifier, sign detector) are outside the active scope.

- [ ] Confirm whether the assignment mandates TensorFlow/Keras
- [ ] Acquire LISA Traffic Light dataset (+ Bosch for small lights)
- [ ] Convert to YOLO format, split train/val/**held-out test**
- [ ] Fine-tune YOLOv8n: 4 classes — red / yellow / green / off
- [ ] Evaluate on held-out test: mAP50, per-class recall, recall by box size
- [ ] Swap into pipeline, export TensorRT, re-run diagnostics on the same clips
- [ ] Write up baseline vs custom, including the FPS cost

**Stretch:** ego-lane segmentation model on TuSimple, replacing the geometric
corridor in `ego_corridor.py`.

---

## Current measured state

| | Frankfurt (city) | BDDA 100 (highway) | BDDA 1003 (urban) |
|---|---|---|---|
| Ego corridor available | 100% | 100% | 100% |
| Corridor centre jitter | 9.8 px/frame | 3.0 px/frame | 30 px/frame |
| Corridor width | 27% of frame | 42% | 13% |
| Green spill off road | 13% | 6% | 23% |
| Vehicles | 10.5/frame | 8.8/frame | 11.6/frame |
| Light state classified | 87% | 96% | 98% |

Traffic lights: below-horizon 0%, state flips 1%, frame presence 38.3%.
Performance: **28-32 FPS** at 720p.

---

## Known limitations carried forward

1. **Dense urban is the weak case** — BDDA 1003 gives 13% corridor width, 23%
   spill, 30 px jitter. The road mask is small and fragmented there.
2. **The corridor is model-based, not detected** — assumes the ego lane is the road
   surface directly ahead, so it cannot represent a lane change in progress.
   Phase 7 stretch goal addresses this.
3. **Light recall capped by YOLOv8n-COCO** on small distant lights (38.3%).
   Phase 7 main goal addresses this.
4. **Light state uses colour opponency** — vulnerable at night and under strong
   backlighting. Phase 7 addresses this.
5. **1080p is CPU-bound** on video decode. 720p is the practical sweet spot.
6. **Light association is by grid cell**, not IoU tracking, so two very close
   lights can merge into one track.

---

## Validation gates (all currently passing)

| Test | Pass condition | Status |
|------|---------------|--------|
| Drivable area detected | >90% of frames | PASS (92.6%) |
| Lane pixels detected | >80% of frames | PASS (91.9%) |
| Vehicles detected | >50% of frames | PASS (92.8%) |
| Light state classified | >70% of detections | PASS (85.6%) |
| Live throughput | >25 FPS at 720p | PASS (28-32) |
| Lane fitter self-checks | 10/10 | PASS |
