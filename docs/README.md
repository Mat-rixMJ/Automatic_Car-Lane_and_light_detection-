# CarLaneI — Documentation

Research-backed documentation for the CarLaneI driving-perception project (lane +
traffic-light + traffic-sign detection). Compiled from a survey of the published
field and mapped to what this project actually does. All external claims are
attributed inline; content was rephrased for licensing compliance.

## Contents

| File | What's inside |
|---|---|
| [`01_field_research.md`](01_field_research.md) | Survey of the state of the art across all three tasks (multi-task perception, lane/drivable, traffic-light state, traffic-sign detection), with datasets, benchmarks, and papers. Grounds every CarLaneI design choice in the literature. |
| [`02_what_success_looks_like.md`](02_what_success_looks_like.md) | An 8-dimension rubric for a *successful* project, a self-assessment scorecard for CarLaneI, the gaps to reach "excellent", and a one-paragraph judge pitch. |
| [`03_frontend_playbook.md`](03_frontend_playbook.md) | How to build a *valuable* frontend, grounded in Uber's AVS/visualization work and HUD/dashboard research. Information architecture, HUD layers, tech choices, and a prioritised upgrade backlog. |
| [`05_gap_analysis_and_improvement_plan.md`](05_gap_analysis_and_improvement_plan.md) | **Compares current CarLaneI to the rubric**, lists what's missing in the perception system and the frontend, and gives a prioritised, phased implementation plan with an effort/impact matrix. |
| [`04_sources.md`](04_sources.md) | Consolidated, categorised list of all sources. |

## How CarLaneI maps to the field (one look)

| Task | Field standard | CarLaneI |
|---|---|---|
| Road geometry | Multi-task net (YOLOP-style) | YOLOP (drivable + lane) |
| Ego lane | Region segmentation more robust than lane-lines on real roads | **Trained** YOLOv8n-seg on BDD100K, mask mAP50 **0.959** |
| Traffic lights | Detect + R/Y/G state, temporal helps | YOLOv8n + colour + **temporal voting** |
| Traffic signs | Real data > synthetic; shape-class reliable, fine names hard | **Trained** YOLOv8n on real GTSDB, 4-class, mAP50 **0.914** |
| Deployment | Quantized / TensorRT edge | 4 models TensorRT FP16, ~20 FPS 720p RTX 3050 |
| Showcase | Interactive web viz for inspection | FastAPI + web app, live HUD, real telemetry |

## Key measured results (held-out)

- **Ego-lane segmentation:** mask mAP50 **0.959**; IoU **0.59 vs 0.06** for the
  prior heuristic corridor (same held-out frames).
- **Traffic-sign detector:** mAP50 **0.914** on real GTSDB val (4 super-classes).
- **Throughput:** **~20 FPS** at 720p on an RTX 3050 6GB (TensorRT FP16).

See the project root `RESULTS.md` for the reproduce commands.
