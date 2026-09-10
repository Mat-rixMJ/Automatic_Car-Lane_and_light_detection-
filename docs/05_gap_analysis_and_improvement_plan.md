# Gap Analysis & Improvement Plan

Compares the **current** CarLaneI against the success rubric (`02_what_success_looks_like.md`)
and the frontend playbook (`03_frontend_playbook.md`), then gives an actionable,
prioritised plan for both the perception system and the web frontend.

Status snapshot at time of writing: ego-lane training at epoch ~43/60 (best mask
mAP50 **0.959**); sign detector mAP50 **0.914**; web app runs upload→process→play
with a live HUD and real telemetry tables.

---

## PART A — Where the project stands (honest scorecard)

| # | Dimension | Target | Current | Gap |
|---|---|---|---|---|
| 1 | Held-out metrics | mAP/IoU on public sets | ego 0.959, sign 0.914, IoU 0.59 | Not reported on the *official* leaderboard splits/protocol |
| 2 | Honesty / limitations | explicit hard cases | documented in RESULTS/docs | Not surfaced *in the UI* |
| 3 | Multi-task architecture | shared encoder + detectors | YOLOP + 3 detectors | ✔ (no major gap) |
| 4 | Edge / real-time | ≥15 FPS, TensorRT | ~20 FPS FP16 | FPS not shown live vs a target line |
| 5 | Real-data training | genuine + held-out | BDD + GTSDB | Lane trained on 8k of 70k available; no night split |
| 6 | Temporal signal robustness | voting / video | geometry gate + voting | No "relevance" (which light is mine); no abnormal states |
| 7 | Interactive showcase | run + visualise | web app + HUD | Missing timeline, charts, before/after, popups, live mode |
| 8 | Reproducibility | scripts + docs | present | No single "one-command eval that regenerates all numbers" |

**Summary:** the *science* is solid and defensible. The biggest, cheapest wins are
now in **presentation/precision** — surfacing what already exists (confidence,
limitations, before/after) and tightening evaluation to official protocols.

---

## PART B — Perception system: what's missing & how to improve

Ordered by impact-to-effort. Each item: *what*, *why*, *how*.

### B1. Finish + lock the lane model (in progress) — **do first**
- **What:** let training reach 60 epochs, then re-copy `best.pt` → `models/ego_seg.pt`
  and re-export the TensorRT engine.
- **Why:** the deployed engine is currently the epoch-40 model; the final is better.
- **How:** `finalize_overnight.py` already automates copy + export + IoU re-eval, or
  run `train_ego_seg.py --resume` to completion then `export`. **~1 command.**

### B2. Report on official evaluation protocol — **highest credibility win**
- **What:** evaluate ego-lane on the BDD100K val split and signs on the GTSDB test
  split using the standard metric definitions, and write the numbers next to
  published baselines.
- **Why:** "0.959 on my held-out split" is good; "X on the same split the SOTA papers
  use" is *comparable* and far more persuasive to judges.
- **How:** extend `eval_lane_iou.py` to also emit mask mIoU over the full val set;
  add a `eval_signs.py` that runs `YOLO.val()` on the GTSDB test split; drop a
  comparison table into `RESULTS.md`.

### B3. Night / adverse-condition evaluation — **addresses the field's hard case**
- **What:** carve a night subset (BDD has time-of-day tags; we already score
  brightness in `select_test_videos.py`) and report lane + sign + light numbers on it.
- **Why:** the literature repeatedly flags night as the failure mode; a night number
  is a strong differentiator.
- **How:** filter BDD val by low brightness, run the eval scripts on that subset.

### B4. Traffic-light: relevance + a dedicated model — **safety precision**
- **What:** (a) mark which detected light applies to the ego lane (relevance);
  (b) optionally fine-tune a light detector on BSTLD/DTLD to lift small-light recall.
- **Why:** COCO's light class caps recall on distant lights; "relevance estimation"
  is an active research sub-problem (TLD-READY). This is the weakest link.
- **How:** relevance = simple heuristic first (light nearest the lane vanishing
  point / above the ego corridor), shown as a highlighted "ACTIVE" light. Dedicated
  model is a larger effort — schedule after the demo.

### B5. Grow the lane training set — **accuracy headroom**
- **What:** train on more of the 70k BDD masks (currently 8k) and add left/right
  flip-consistent augmentation.
- **Why:** more data → better generalisation, especially urban/occluded.
- **How:** `prepare_ego_seg.py --n-train 30000`; retrain. GPU-time bound, not code.

### B6. One-command reproducibility — **methodology polish**
- **What:** a `make_report.py` (or `.bat`) that runs all eval scripts and regenerates
  every number in `RESULTS.md`.
- **Why:** judges trust numbers a script can regenerate.
- **How:** thin wrapper calling the existing eval scripts, writing a dated report.

### B7. Ablation table — **shows engineering rigour**
- **What:** one table quantifying each component's contribution (corridor vs ego-seg;
  with/without temporal voting; synthetic vs real sign detector).
- **Why:** demonstrates that each design decision was measured, not guessed.
- **How:** most numbers already exist across prior runs; collect into one figure.

---

## PART C — Frontend: missing pieces & how to make it more interactive

The backend already returns rich per-frame telemetry (offset, LDW, per-frame
vehicles/signs/lights with confidence, dominant signal). The UI currently uses only
a fraction of it. **Most frontend wins are "surface data we already have."**

Ordered by impact-to-effort.

### C1. Signal + LDW timeline strip under the video — **highest impact**
- **What:** a horizontal bar spanning the clip, coloured by dominant light state
  (red/amber/green/grey) over time, with LDW markers; click to seek the video.
- **Why:** the HUD literature values *continuous* feedback; a timeline turns 179
  frames of telemetry into one glanceable story and makes the video scrubbable by
  event.
- **How:** render a row of `<div>` segments from `telemetry.frames[i].dominant_light`;
  click sets `video.currentTime`. Pure CSS/JS, data already present.

### C2. Before/after toggle (heuristic corridor vs learned ego-seg) — **proves the win**
- **What:** a switch that plays the same clip processed two ways, or overlays both
  masks, so the IoU **0.06 → 0.59** gain is *seen*.
- **Why:** the single most convincing thing for judges — the improvement becomes visual.
- **How:** backend: add a `mode=corridor|egoseg` param to `/api/process` (pipeline
  already has both code paths via the `ego_seg` fallback). Frontend: a toggle that
  requests/plays the chosen output. Medium effort, very high payoff.

### C3. Live charts panel — **surfaces the "reasoning"**
- **What:** two small line charts — lane-offset trace and a light-state timeline —
  plus a per-frame mean-confidence readout, all synced to playback.
- **Why:** dashboards should show the decision process, not just outputs.
- **How:** a tiny inline SVG/canvas sparkline drawn from `telemetry.frames`
  (offset, confidences). No heavy chart lib needed.

### C4. Detection pop-ups / toasts + alignment polish — **interactivity + UX**
- **What (pop-ups):** when an LDW ("LANE DEPARTURE"/"drifting") fires or the dominant
  signal turns RED during playback, show a transient toast/banner. On hovering a
  table row, pop a small card with that class's confidence and a sample thumbnail.
- **What (alignment):** normalise card grid gaps, align the metric strip and hero
  box baselines, make the two event tables equal-height, and give the workbench a
  consistent 12-col rhythm so nothing looks ragged.
- **Why:** pop-ups make the page feel alive and draw the eye to safety events;
  alignment is what separates "student project" from "polished".
- **How:** a small toast component (fixed-position div, auto-dismiss) driven by the
  HUD loop when `ldw`/`dominant_light` changes; CSS grid/gap normalisation pass.

### C5. Confidence surfacing everywhere — **credibility**
- **What:** show per-detection confidence in the HUD tooltip and a per-frame mean;
  colour-code low-confidence detections.
- **Why:** the field emphasises reliability/uncertainty; showing confidence signals maturity.
- **How:** data already in `telemetry.frames[i].{vehicles,signs,lights}[].conf`.

### C6. SOTA comparison card — **context for the numbers**
- **What:** a static, sourced table putting CarLaneI's mAP/IoU next to published
  BDD100K/GTSDB baselines (from `01_field_research.md`).
- **Why:** numbers mean more with a reference point.
- **How:** static HTML section; cite sources.

### C7. Live mode (webcam / RTSP via WebSocket) — **aspirational, do last**
- **What:** stream frames from a webcam, process, and push annotated frames + telemetry
  over a WebSocket for true real-time.
- **Why:** "it runs live" is the ultimate demo, but it's the most work and least
  necessary for a judged showcase.
- **How:** FastAPI WebSocket endpoint; a frame loop; MJPEG/base64 frames to a canvas.

### C8. Robustness & accessibility pass
- **What:** graceful error toasts if the backend is down or a job fails; keyboard
  video controls; colour-blind-safe state palette (icons+text, not colour alone);
  mobile layout check.
- **Why:** avoids a live-demo failure and broadens who can read it.

---

## PART D — Recommended order of execution

**Phase 1 — lock the science (today):**
1. B1 finish + export final lane model.
2. B2 official-protocol eval + comparison table in `RESULTS.md`.

**Phase 2 — make the demo undeniable (highest judge impact):**
3. C1 signal/LDW timeline strip.
4. C2 before/after toggle (visualises the IoU win).
5. C4 pop-ups/toasts + alignment polish.

**Phase 3 — depth & credibility:**
6. C3 charts + C5 confidence surfacing.
7. B3 night eval, B7 ablation table, C6 SOTA comparison card.

**Phase 4 — stretch:**
8. B4 light relevance + dedicated model, B5 bigger lane set, C7 live mode.

---

## PART E — Effort/impact matrix (quick reference)

| Item | Effort | Judge impact | Phase |
|---|---|---|---|
| B1 finalise lane model | XS | High | 1 |
| B2 official eval + table | S | High | 1 |
| C1 timeline strip | S | High | 2 |
| C2 before/after toggle | M | Very high | 2 |
| C4 pop-ups + alignment | S | High | 2 |
| C3 charts + C5 confidence | M | Medium | 3 |
| B3 night eval | S | Medium-High | 3 |
| B7 ablation table | S | Medium | 3 |
| C6 SOTA card | XS | Medium | 3 |
| B4 light relevance/model | L | Medium | 4 |
| B5 bigger lane set | M (GPU) | Medium | 4 |
| C7 live mode | L | High (wow) | 4 |

---

## One-line recommendation

The perception science is already judge-ready; **spend the next effort on Phase 1
(finalise + official numbers) and Phase 2 (timeline, before/after toggle, pop-ups &
alignment)** — that combination turns a solid system into a demo that visibly proves
its own claims.
