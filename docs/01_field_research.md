# Field Research — Lane, Traffic-Light & Traffic-Sign Perception

A survey of the research landscape for the three tasks this project combines, with
sources. Compiled to ground CarLaneI's design decisions in the published
state of the art. All external claims are attributed inline; content is
paraphrased/summarised for licensing compliance.

> Content was rephrased for compliance with licensing restrictions.

---

## 1. The big picture: why these three tasks are usually combined

Camera-based perception is the foundation that lets a vehicle understand its
surroundings for safe driving, drawing on recent computer-vision advances
([*Applications of Computer Vision in Autonomous Vehicles*, arXiv 2311.09093](https://arxiv.org/html/2311.09093v3)).
Detecting and recognising road lanes, traffic signs, and traffic lights is
described as an essential and long-standing ADAS research problem within intelligent
transport systems
([*Navigation System for Autonomous Vehicle: A Survey*, ResearchGate](https://www.researchgate.net/publication/348370339_Navigation_System_for_Autonomous_Vehicle_A_Survey)).

The dominant modern architecture for the road-geometry part is **multi-task /
"panoptic" perception**: one shared encoder feeds several task-specific decoder
heads, so lane, drivable-area, and object tasks are solved together to cut compute
and inference time while improving each task
([YOLOP, *You Only Look Once for Panoptic Driving Perception*, arXiv 2108.11250](https://arxiv.org/html/2108.11250v2);
[hustvl/YOLOP GitHub](https://github.com/hustvl/YOLOP)).

**Relevance to CarLaneI:** the project uses YOLOP (the drivable-area/lane
multi-task network) as its road-geometry backbone, and adds detection for objects,
lights, and signs — mirroring the standard split in the literature between a
segmentation multi-task net and dedicated detectors.

---

## 2. Lane detection & drivable area

### State of the art
- **YOLOP** (Wu et al., MIR 2022) established the multi-task pattern — one encoder,
  three decoders (object detection, drivable-area segmentation, lane-line
  segmentation) — reporting state-of-the-art accuracy and speed on the challenging
  **BDD100K** dataset
  ([arXiv 2108.11250](https://arxiv.org/html/2108.11250v2)).
- **YOLOPv2 / "Better, Faster, Stronger"** pushed the same three-task formulation to
  new SOTA on BDD100K
  ([arXiv 2208.11434](https://arxiv.org/html/2208.11434)).
- **Q-YOLOP** added quantization-aware training and reported mAP@0.5 ≈ 0.622 for
  detection and mIoU ≈ 0.612 for segmentation while keeping compute/memory low for
  embedded deployment
  ([arXiv 2307.04537](https://arxiv.org/html/2307.04537)).
- **TriLiteNet** is a recent lightweight multi-task model: reported ~85.6% vehicle
  recall, ~92.4% mIoU drivable-area, and ~82.3% lane-line accuracy with only ~2.35M
  parameters and ~7.72 GFLOPs
  ([arXiv 2509.04092](https://arxiv.org/html/2509.04092)).

### Key benchmark
- **BDD100K** — the standard large-scale driving dataset used across these papers
  for lane, drivable-area, and object tasks
  ([bdd100k.com](https://www.bdd100k.com/)).

### Two lane paradigms (and where CarLaneI sits)
1. **Lane-line detection** — find the painted line instances (segmentation or
   row-classification, e.g. Ultra-Fast-Lane-Detection). Fragile when lines are
   dashed, occluded, or absent.
2. **Drivable-area / ego-lane segmentation** — segment the region you can drive on.
   More robust when markings are missing.

CarLaneI evaluated a pretrained lane-line model (UFLD) first, found it failed on
this footage (a domain/geometry gap), and switched to **training an ego-lane
segmentation model on BDD100K "direct drivable" masks** — the region-based
paradigm, which the literature shows is the more reliable signal on real,
marking-sparse roads.

---

## 3. Traffic-light detection & state recognition

### Problem definition
Traffic-light recognition = detect the light's bounding box **and** classify its
state (red / yellow / green, sometimes "off"). Formally: given an image, output
boxes and class labels for each light instance
([*Recognizing Traffic Lights with Deep Learning*, Bomberbot guide](https://www.bomberbot.com/data-science/recognizing-traffic-lights-with-deep-learning-a-comprehensive-guide/)).
It is called a critical component of the AV perception stack, essential at SAE
Level 3+ autonomy
([TLD-READY, arXiv 2409.07284](https://arxiv.org/pdf/2409.07284v1)).

### Datasets
- **Bosch Small Traffic Lights Dataset (BSTLD)** — 8,334 images labelling signals
  as off / green / red / yellow
  ([Camera-based Context-aware TLD, ML4AD 2023](https://ml4ad.github.io/files/papers2023/Camera-based%20Context-aware%20Traffic%20Light%20Detection%20for%20Self-Driving%20Vehicles.pdf)).
- **DriveU Traffic Light Dataset (DTLD)** and others extend scale and diversity.

### Recent results & themes
- A 2025 CNN method reported precision ~0.992 (red), ~0.995 (yellow), ~0.853
  (green) and mAP@0.5 ~0.947, i.e. **green is consistently the hardest state**
  ([MDPI WEVJ 16/8/441](https://www.mdpi.com/2032-6653/16/8/441/xml)).
- **CSDETR** (improved RT-DETR) notes most methods handle only R/Y/G and struggle
  to balance accuracy vs speed, and that abnormal states are often neglected
  ([Springer 11554-026-01864-6](https://link.springer.com/10.1007/s11554-026-01864-6)).
- **Temporal / video-based** recognition beats single-frame under occlusion and
  bad lighting
  ([Rockchip RV1126 TLR, arXiv 2503.23965](https://arxiv.org/html/2503.23965v1)).
- **Relevance estimation** — deciding *which* light applies to the ego vehicle — is
  an active sub-problem
  ([TLD-READY, arXiv 2409.07284](https://arxiv.org/pdf/2409.07284v1)).

**Relevance to CarLaneI:** the project detects lights with a COCO detector and
classifies state with colour opponency, then adds a **geometry gate + temporal
state voting** — directly addressing the two themes above (single-frame fragility
and relevance/false positives). Its own measurements show the temporal gate cuts
sub-horizon false positives and state flicker substantially.

---

## 4. Traffic-sign detection & recognition

### Datasets (small → large)
- **GTSRB** — German classification benchmark, 43 classes of cropped signs.
- **GTSDB** — German *detection* benchmark, full road images with boxes.
- **LISA TS**, **TT100K** (Chinese, dense), **CCTSDB**, **BelgiumTS**.
- **Mapillary Traffic Sign Dataset (MTSD)** — described as the largest and most
  diverse, global imagery with fine-grained classes and strong detection +
  classification baselines
  ([arXiv 1909.04422](https://arxiv.org/html/1909.04422v2)).
  A benchmarking study uses LISA-TS, GTSD, TT100K and MTSD together
  ([Evaluating & Benchmarking OD Models for TS/TL, ACCV 2022 W](https://openaccess.thecvf.com/content/ACCV2022W/DLSOD/papers/Mishra_Evaluating_and_Bench-marking_Object_Detection_Models_for_Traffic_Sign_and_ACCVW_2022_paper.pdf)).

### Recent results & themes
- **YOLO-family detectors** dominate real-time sign detection; recent work adds
  optimized receptive fields and anchor-free fusion for accuracy
  ([arXiv 2410.17144](https://arxiv.org/html/2410.17144v1)).
- **Low-light is a distinct hard case**: YOLO-LLTS reports gains specifically on
  night splits (GTSDB-night, TT100K-night, CCTSDB2021)
  ([arXiv 2503.13883](https://arxiv.org/abs/2503.13883)).
- **Vision Transformers** are emerging for classification speed+accuracy on GTSRB /
  BelgiumTS
  ([arXiv 2404.19066](https://arxiv.org/html/2404.19066v1)).
- **Robustness under visual degradation** (blur, weather) and **interpretability**
  are stressed because misclassification directly affects safety decisions
  ([MDPI Algorithms 19/7/557](https://www.mdpi.com/1999-4893/19/7/557/xml)).
- A GTSDB detector detecting 29 sign kinds reported ~73.9% classification accuracy,
  showing fine-grained GTSDB detection is genuinely hard
  ([FSADD, MDPI Electronics 12/3/725](https://www.mdpi.com/2079-9292/12/3/725)).

**Relevance to CarLaneI — the honest design choice.** The project first trained a
GTSRB 43-class classifier and found it *saturated* (over-confident on
out-of-distribution signs), and a detector trained on *synthetic* paste-ups that
generalised poorly. It then retrained a **single-stage detector on real GTSDB road
images** collapsed to **4 shape/colour super-classes** (prohibitory / mandatory /
danger / other). This matches the literature's finding that fine-grained naming is
the hard, error-prone part, while shape/colour class is reliable — a defensible
accuracy-vs-honesty trade-off.

---

## 5. Where this whole area is heading

- **Unified multi-task / multi-modal frameworks** are expanding beyond road geometry
  to driver state and traffic context (e.g. MMTL-UniAD, PRISM-MTL)
  ([arXiv 2504.02264](https://arxiv.org/html/2504.02264v1);
  [MDPI Mathematics 14/15/2812](https://www.mdpi.com/2227-7390/14/15/2812)).
- **Edge deployment** (quantization, lightweight backbones, TensorRT) is a first-
  class concern, not an afterthought
  ([Q-YOLOP](https://arxiv.org/html/2307.04537); [TriLiteNet](https://arxiv.org/html/2509.04092)).
- **Reliability & safety framing** — ADAS is still maturing and needs continuous
  optimisation; perception errors have direct safety impact
  ([MDPI Sensors 24/19/6223](https://www.mdpi.com/1424-8220/24/19/6223);
  [*A Reliable Perception Framework*, arXiv 2504.19722](https://arxiv.org/html/2504.19722v1)).

**Takeaway for CarLaneI:** the project is aligned with the field — a multi-task
segmentation backbone, dedicated detectors, temporal stabilisation for signals, a
real-data honest sign detector, and TensorRT edge deployment on a laptop GPU.
