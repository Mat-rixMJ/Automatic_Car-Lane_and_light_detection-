# Frontend Playbook — Building a Valuable Perception Showcase

How to turn CarLaneI into a frontend that convinces technical reviewers, grounded
in how the industry builds autonomous-vehicle visualization and HUD interfaces.
Sources attributed inline; external content paraphrased for compliance.

> Content was rephrased for compliance with licensing restrictions.

---

## 1. Why the frontend matters (evidence)

Perception is a visual problem, and the people who build it invest heavily in
visualization. Uber's ATG built a dedicated **web-based data-visualization platform**
for exploring, inspecting, and debugging how their self-driving vehicles perceive
the world
([*Engineering Uber's Self-Driving Car Visualization Platform for the Web*](https://www.uber.com/blog/atg-dataviz)).
They then open-sourced **AVS** (Autonomous Vehicle Visualization) — a web-first
standard and toolkit for describing and visualising perception, motion, and
planning data, explicitly to help teams *make development decisions* from that data
([*Introducing AVS*, Uber](https://www.uber.com/us/en/blog/avs-autonomous-vehicle-visualization/)).

Two lessons for us:
1. A good perception frontend is a **decision-making and inspection tool**, not
   decoration.
2. The web is the right delivery target — it's what Uber chose for reach and
   interactivity.

For in-vehicle-style overlays, HUD research shows that continuous visual feedback
(e.g. a risk/lane profile on a head-up display) helps drivers build a clear mental
model and maintain situational awareness, versus intermittent alerts
([*Continuous Visual Feedback of Risk*, arXiv 2301.10933](https://arxiv.org/html/2301.10933v1);
[*Design of HUD interfaces for automated vehicles*, Zenodo](https://zenodo.org/records/7994492/files/Design%20of%20head-up%20display%20interfaces%20for%20automated%20vehicles.pdf)).

---

## 2. Principles (what "valuable" means here)

From the dashboard/telemetry and AV-UI literature:

- **Data shape first.** A clean interface can't rescue badly structured data; how
  the data is filtered and delivered to the screen is decided before visual design
  ([*Designing Dashboards for Smart-Device Data*, Made by Shape](https://madebyshape.co.uk/web-design-blog/building-the-dashboard-designing-web-interfaces-for-smart-device-data/)).
  → *We already emit a structured telemetry JSON per run; build the UI on top of it.*
- **Dynamic beats static.** A user study of teleoperation GUIs found the dynamic
  interface significantly outperformed the static one in usability and task time,
  both scoring well on the System Usability Scale (SUS)
  ([arXiv 2504.21563](https://arxiv.org/abs/2504.21563)).
  → *Live HUD synced to playback > a frozen annotated image.*
- **Show the reasoning, not just the output.** Telemetry dashboards are advised to
  capture the decision process — the steps and the "why"
  ([*Building Your First AI Agent Telemetry Dashboard*](https://clawpulse.hashnode.dev/building-your-first-ai-agent-telemetry-dashboard-a-real-time-monitoring-guide)).
  → *Surface confidence, the temporal-voting decision, lane offset — not only boxes.*
- **Dual audience.** Engineering UIs serve both live in-field control and
  after-the-fact analysis/parameterisation
  ([*Beyond the Dashboard*, Springer](https://link.springer.com/chapter/10.1007/978-3-032-30427-8_10)).
  → *Offer a "run a clip" demo mode AND a "metrics/analysis" mode.*

---

## 3. The information architecture that works for CarLaneI

A perception showcase should answer four questions, in order:

1. **Is it credible?** → hero + measured metric cards (held-out mAP/IoU, FPS).
2. **What does it do to a clip?** → the interactive workbench: pick/upload → run →
   annotated video + live HUD.
3. **What did it actually detect?** → event registry tables populated from the real
   run (sign classes, light-state distribution, vehicles/frame).
4. **How is it built / why trust it?** → model-stack cards + honest limitations.

CarLaneI's current page already follows this order. Keep it.

---

## 4. The HUD: what to overlay (grounded in the field)

The overlay is where perception work earns trust. Recommended layers, matching how
AV viz tools and HUDs present state:

| Layer | Why | CarLaneI source |
|---|---|---|
| Ego-lane fill + outline | The core lane output; region-based reads clearly | ego-seg mask |
| Vehicle boxes + class + conf | Object awareness | YOLOv8n |
| Traffic-light boxes coloured by state | Signal state is the safety-critical bit | detector + colour + voting |
| Sign boxes coloured by class | Sign awareness | GTSDB detector |
| Lane-offset / LDW banner | Continuous lateral feedback (HUD principle) | ego offset ratio |
| Live counters (FPS, cars/lights/signs) | System-health at a glance | telemetry per frame |
| Dominant-signal chip | The single "what should I do" signal | temporal vote |

Keep colour semantics consistent (red/amber/green = signal states; one accent for
the ego lane) so the eye parses it instantly — a core HUD readability point.

---

## 5. Recommended tech (pragmatic vs aspirational)

**Pragmatic (what CarLaneI uses — right for a demo):**
- FastAPI backend wrapping the real pipeline; returns annotated MP4 + telemetry JSON.
- Vanilla HTML/CSS/JS with Tailwind (CDN) — no build step, instant to run.
- `<video>` element for playback; a `requestAnimationFrame` loop maps
  `currentTime × fps` to the telemetry frame for the live HUD.
- One-click launcher (`start_webapp.bat`).

This is the correct level for a judged showcase: zero setup, real data, runs on the
same GPU.

**Aspirational (if this became a product / research tool):**
- **Uber AVS / streetscape.gl + deck.gl** for a full spatial visualization (bird's-
  eye, 3D scene, timeline scrubber over logged data)
  ([AVS](https://www.uber.com/us/en/blog/avs-autonomous-vehicle-visualization/); [deck.gl](https://deck.gl)).
- A React/Svelte app with a proper timeline, per-frame stepping, and side-by-side
  before/after (heuristic vs learned) toggles.
- WebSocket streaming for true live (webcam / RTSP) instead of upload→process→play.
- Charts (confidence over time, state timeline, offset trace) via a lightweight
  charting lib.

---

## 6. Concrete upgrade backlog for CarLaneI's frontend

Prioritised, each independently shippable:

1. **Timeline / state-strip** under the video: a horizontal bar coloured by dominant
   signal state over time, clickable to seek. (High impact, matches HUD "continuous
   feedback" principle.)
2. **Before/after toggle** — play the same clip with the old heuristic corridor vs
   the trained ego-seg, so the IoU-0.06→0.59 gain is *seen*, not just stated.
3. **Confidence surfacing** — show per-detection confidence and a per-frame mean, per
   the "show the reasoning" principle.
4. **Charts panel** — lane-offset trace and light-state timeline from the telemetry
   JSON (data already there).
5. **Live mode** — WebSocket frame streaming from a webcam for a true real-time demo.
6. **Comparison to SOTA** — a small static table putting CarLaneI's numbers next to
   published BDD100K/GTSDB baselines (credibility).
7. **Accessibility** — keyboard controls, ARIA labels, colour-blind-safe state
   palette (don't rely on red/green alone; add icons/text).

---

## 7. Anti-patterns to avoid (learned the hard way here)

- **Fabricated numbers.** The original design mockup shipped invented specs (a
  specific edge chip, 120+ sign classes, sub-10ms latency, spline lane fitting).
  Against a real demo those *hurt* credibility. Every figure must be measured. ✅ Fixed.
- **Static hero with no interaction.** A frozen image reads as a poster; reviewers
  reward "let me run it." ✅ Addressed via the workbench.
- **Overlay soup.** Too many layers at once is unreadable — keep consistent colours
  and let the user toggle layers.
- **Hiding failure cases.** Show a hard clip too; honesty is a trust multiplier
  (the field's reliability framing).

---

## 8. Summary

The literature is consistent: the value of a perception frontend is that it turns
opaque model outputs into something a human can *inspect, trust, and make decisions
from*. CarLaneI's web app already hits the essentials — interactive run, live HUD,
real telemetry tables, honest metrics. The backlog in §6 (timeline, before/after
toggle, confidence, charts, live mode, SOTA comparison) is the path from a solid
showcase to a genuinely valuable inspection tool.
