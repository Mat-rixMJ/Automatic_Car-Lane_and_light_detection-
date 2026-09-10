# What a Successful Project Looks Like

A concrete rubric for a lane + traffic-light + traffic-sign perception project,
drawn from the published field (see `01_field_research.md`) and framed for a
technical-judge audience. Each dimension lists what "good" looks like, then where
**CarLaneI** currently stands — honestly.

---

## The 8 dimensions judges actually weigh

### 1. Real, defensible metrics (not demo screenshots)
**Good:** every claim is a number on *held-out* data, with the dataset named and
the metric standard (mAP@0.5, mIoU, IoU, precision/recall). SOTA papers always
report on public benchmarks (BDD100K, GTSDB, BSTLD) this way.

**CarLaneI:**
- Ego-lane segmentation: **mask mAP50 0.959** on held-out BDD100K.
- Ego-lane IoU **0.59 vs 0.06** for the old heuristic (measured, same held-out frames).
- Sign detector: **mAP50 0.914** on real GTSDB val (per-class: prohibitory 0.972,
  danger 0.974, mandatory 0.858, other 0.852).
- Throughput: **~20 FPS at 720p** on an RTX 3050 (TensorRT FP16).

✅ Strong. This is the single biggest differentiator vs a mockup.

### 2. Honesty about limitations
**Good:** the field stresses reliability and safety; over-claiming is a red flag.
Papers explicitly report the hard cases (green lights, night signs, occlusion).

**CarLaneI:**
- Documents that the sign detector is GTSDB/European (won't fire on US signage),
  reports 4 shape-classes not specific names, and that it *removed* a saturated
  classifier rather than ship it.
- Documents the lane occlusion failure case (bumper-to-bumper) plainly.

✅ Strong — this is a genuine trust signal for reviewers.

### 3. Multi-task / efficient architecture
**Good:** shared-encoder multi-task perception (YOLOP-style) for road geometry,
dedicated detectors for objects/lights/signs, and temporal stabilisation for
signals — the standard modern stack.

**CarLaneI:** YOLOP (drivable/lane) + YOLOv8n (vehicles/lights) + trained ego-seg +
trained sign detector + temporal light voting. ✅ Matches the field.

### 4. Edge / real-time deployment
**Good:** quantization / TensorRT / lightweight backbones are first-class in recent
work (Q-YOLOP, TriLiteNet). Real-time = ≥ ~15–30 FPS.

**CarLaneI:** all four models TensorRT-compiled (FP16), ~20 FPS at 720p on a 6GB
laptop GPU. ✅ Solid for the hardware class.

### 5. Trained on real data, methodology sound
**Good:** trained on genuine annotated data, held-out split, no leakage, sensible
augmentation. The literature warns synthetic-only training generalises poorly.

**CarLaneI:** ego-seg trained on 8k real BDD masks; sign detector retrained on real
GTSDB after the synthetic version underperformed. ✅ The "synthetic → real" pivot
is itself a strong methodology story.

### 6. Temporal robustness for signals
**Good:** video/temporal methods beat single-frame under occlusion and lighting;
relevance estimation (which light is mine) matters.

**CarLaneI:** geometry gate + temporal state voting; measured reduction in
sub-horizon false positives and state flicker. ✅ Directly on-theme.

### 7. A showcase people can interact with
**Good:** perception work is visual; the strongest demos let a reviewer *run it*
and *see* the overlays + live telemetry, not just read a table. (Uber built a whole
web visualization platform for exactly this — see `03_frontend_playbook.md`.)

**CarLaneI:** FastAPI backend + web frontend — upload/pick a clip, run the real
pipeline, watch the annotated video with a live HUD and real detection tables.
✅ In place; the differentiator over a static poster.

### 8. Reproducibility
**Good:** scripts to prepare data, train, export, evaluate, and run; documented
commands; pinned environment.

**CarLaneI:** `prepare_*`, `train_*`, `export_tensorrt`, `eval_lane_iou`,
`validate_pipeline`, `run_pipeline_fast`, plus `run.bat` / `start_webapp.bat`.
✅ Present.

---

## Scorecard (self-assessment)

| Dimension | Target | CarLaneI | Status |
|---|---|---|---|
| Real held-out metrics | mAP/IoU on public sets | ego 0.959, sign 0.914, IoU 0.59 | ✅ |
| Honesty / limitations | explicit hard cases | documented | ✅ |
| Multi-task architecture | shared encoder + detectors | YOLOP + 3 detectors | ✅ |
| Edge / real-time | ≥15 FPS, TensorRT | ~20 FPS, FP16 | ✅ |
| Real-data training | genuine + held-out | BDD + GTSDB | ✅ |
| Temporal signal robustness | voting / video | geometry gate + voting | ✅ |
| Interactive showcase | run + visualise | web app | ✅ |
| Reproducibility | scripts + docs | present | ✅ |

---

## Gaps / next steps to reach "excellent"

Honest list of what would push it from "solid" to "outstanding":

1. **Report on the official public leaderboards' splits** (BDD100K val for lane,
   GTSDB test for signs) in the exact protocol, so numbers are directly comparable
   to published SOTA.
2. **Night / adverse-weather evaluation** — the literature flags this as the hard
   case; a night-split number would be persuasive.
3. **Traffic-light relevance estimation** — mark which light applies to the ego
   lane (active research topic), not just detect all lights.
4. **A dedicated traffic-light model** — COCO's light class caps recall on small
   distant lights; a BSTLD/DTLD-trained detector would raise it.
5. **Per-frame confidence/uncertainty** surfaced in the UI, matching the field's
   emphasis on reliability.
6. **Ablation table** — quantify each component's contribution (e.g. corridor vs
   ego-seg, with/without temporal voting) in one figure.

---

## The one-paragraph pitch (for judges)

> CarLaneI is an edge driving-perception system that detects the ego lane, traffic
> signs, and traffic-signal state in real time (~20 FPS, 720p, RTX 3050). It follows
> the modern multi-task pattern — a YOLOP backbone for road geometry plus dedicated
> detectors — and improves two components with models trained on real public data:
> an ego-lane segmenter (BDD100K, mask mAP50 0.959, IoU 0.59 vs 0.06 for the prior
> heuristic) and a single-stage traffic-sign detector (real GTSDB, mAP50 0.914). Every
> number is measured on held-out data, limitations are stated plainly, and the whole
> thing runs in a browser so you can upload a clip and watch it work.
