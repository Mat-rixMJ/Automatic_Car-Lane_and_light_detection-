# Goal — CarLaneI Build Checklist

Track progress. Each item is either ✅ done, 🔄 in progress, or ⬜ pending.

---

## Phase 1: Environment & Setup

- [x] Project structure created
- [x] Virtual environment (.venv) created
- [x] PyTorch 2.13 + CUDA installed in .venv
- [x] OpenCV, Ultralytics, ONNX Runtime, Transformers installed
- [x] TensorFlow removed — pure PyTorch stack
- [x] Config files (pipeline_config.yaml, german_signs.json, indian_signs.json)
- [x] All source modules parse and import correctly
- [x] memory.md for global package reference

---

## Phase 2: Data & Models

- [x] Download GTSRB dataset (~300 MB, 43 classes, 39K train + 12.6K test images)
- [x] Train sign classifier CNN on GTSRB → **96.81% on official unseen test set**
- [x] Download YOLOv8n pretrained (COCO — class 9 = traffic light confirmed)
- [x] YOLOP loads via PyTorch Hub (lane + drivable area + vehicles)
- [x] Verify sign classifier inference (99.15% train-split, 96.81% official test)
- [x] Verify YOLOv8n traffic light detection on sample image
- [x] Verify YOLOP lane + drivable area on sample image

---

## Phase 3: Pipeline Integration

- [ ] Run `pipeline.py` on a dashcam video — all modules active
- [ ] YOLOP draws lanes + drivable area + vehicle boxes
- [ ] Sign detector finds and classifies signs
- [ ] Traffic light detector finds lights + classifies state (R/Y/G)
- [ ] ADAS dashboard renders (FCW gauge, LDW indicator, speed sign, light icon)
- [ ] FPS counter shows real-time performance (target: 15+ FPS on RTX 3050)

---

## Phase 4: CARLA Integration

- [ ] CARLA installed and running
- [ ] `carla_bridge.py` connects, spawns vehicle, attaches camera
- [ ] Observe mode works — autopilot drives, pipeline annotates
- [ ] Control mode works — pipeline drives the car:
  - [ ] Brakes on red light
  - [ ] Brakes on collision warning
  - [ ] Steers on lane departure
  - [ ] Maintains speed limit from sign detection
- [ ] Record demo video from CARLA (output/carla_demo.mp4)

---

## Phase 5: Depth & ADAS Polish

- [ ] Depth Anything V2 loads and produces depth map
- [ ] FCW uses depth for distance estimation (not just box size)
- [ ] Lane departure warning triggers correctly
- [ ] Speed limit sign read → displayed on dashboard
- [ ] All ADAS alerts visible in the HUD

---

## Phase 6: Demo & Documentation

- [ ] Record 60-second demo video (dashcam or CARLA)
- [ ] Training curves plotted (sign classifier accuracy/loss)
- [ ] README final with screenshots/architecture diagram
- [ ] Code commented and clean
- [ ] Git repo initialized and pushed

---

## Stretch Goals (India Adaptation)

- [ ] Download Indian traffic sign dataset
- [ ] Transfer learning: freeze conv layers, retrain classifier head
- [ ] Add India-specific sign classes (speed breaker, no honking, horn OK)
- [ ] Tune lane detection for Indian road conditions (faded markings)
- [ ] Test on Indian dashcam footage

---

## Current Focus

**Phase 3 → Pipeline integration. Wire all models together, run on video.**

Once Phase 2 is validated, everything else is integration and testing.

---

## Validation Criteria (How we know it works)

| Test | Pass condition |
|------|---------------|
| Sign classifier | >93% accuracy on GTSRB test set |
| YOLOP inference | Lane mask + drivable mask non-empty on road image |
| Traffic light | Detects at least 1 light in a traffic scene image |
| Full pipeline | Runs at >10 FPS on 720p video, no crashes |
| CARLA control | Car brakes within 2s of red light appearing |
| Depth FCW | Warning triggers when vehicle ahead fills >30% of frame width |
