"""Merged Indian+German sign detector - GOOGLE COLAB continuation.

Continues from the LOCAL epoch-20 checkpoint (sign_last_ep20.pt) so no progress
is lost, then trains ~30 more epochs on Colab's T4. Frees the local machine.

Why "load weights + train more" instead of resume=True: the local checkpoint has
a Windows path (d:\\carLane\\...) baked in; Ultralytics resume would choke on it
under Linux. Loading the weights and training fresh transfers all learned
progress without the path fragility.

HOW TO RUN (each block = its own Colab cell):

  # CELL 1 - GPU + install
  !nvidia-smi -L
  !pip -q install ultralytics==8.4.120 kaggle

  # CELL 2 - Kaggle creds (to download GTSDB + Indian sign sets)
  import os
  os.environ['KAGGLE_USERNAME'] = 'matrixmj'
  os.environ['KAGGLE_KEY'] = 'PASTE_KEY'
  print('ok', os.environ['KAGGLE_USERNAME'])

  # CELL 3 - upload the local checkpoint sign_last_ep20.pt
  from google.colab import files
  files.upload()      # pick sign_last_ep20.pt (23MB) from d:\\carLane\\

  # CELL 4 - (optional) mount Drive so the model survives disconnect
  from google.colab import drive
  drive.mount('/content/drive')

  # CELL 5 - paste THIS whole file and run

Output -> /content/signs_merged_detector.pt (+ Drive copy if mounted).
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ------------------------------------------------------------------ config
IMGSZ = 640
BATCH = 16              # 1x T4
EPOCHS_MORE = 30        # train this many MORE epochs on top of the local 20
SEED = 0
VAL_FRAC = 0.15
RUN_NAME = "sign_detector_colab"
OUT_NAME = "signs_merged_detector.pt"
CKPT_UPLOAD = Path("/content/sign_last_ep20.pt")   # the file you upload in CELL 3

WORK = Path("/content")
DS = WORK / "signs_merged"
DL_G = WORK / "dl_gtsdb"
DL_I = WORK / "dl_indian"
DRIVE_OUT = Path("/content/drive/MyDrive/carlane_models")

GTSDB_SLUG = "sovitrath/gtsdb-dataset-in-pascal-voc-structure"
INDIAN_SLUG = "akbaralibatti/indian-traffic-sign-yolo11"
SUPERCLASS_NAMES = ["prohibitory", "mandatory", "danger", "other"]

# --- German GTSDB 43 fine -> 4 super (same mapping as the local prep) ---
FINE_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]
FINE_INDEX = {n: i for i, n in enumerate(FINE_NAMES)}

def gtsdb_super(fid):
    PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
    MANDATORY = {33, 34, 35, 36, 37, 38, 39, 40}
    DANGER = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    if fid in PROHIBITORY: return 0
    if fid in MANDATORY: return 1
    if fid in DANGER: return 2
    return 3

# --- Indian class NAME -> super by shape/colour keyword (same as local) ---
def indian_super(name):
    n = name.lower().strip()
    danger_kw = ["warning","caution","curve","bend","slippery","narrow","hump",
                 "bump","dip","ripple","gap","cattle","pedestrian","school",
                 "children","cross","round about","roundabout ahead","junction",
                 "gap in median","falling","cycle cross","animal","men at work",
                 "road work","barrier","ferry","ford","steep","descent","ascent",
                 "hairpin","loose","guard"]
    mandatory_kw = ["compulsory","mandatory","ahead only","turn right","turn left",
                    "keep left","keep right","go straight","sound horn",
                    "cycle track","pedestrian only","bus","direction"]
    prohib_kw = ["no ","prohibited","restriction","speed limit","no entry",
                 "overtaking prohibited","horn prohibited","no parking",
                 "no stopping","u turn prohibited","no left","no right",
                 "axle load","width limit","height limit","length limit",
                 "load limit","max speed","speed"]
    for kw in danger_kw:
        if kw in n: return 2
    for kw in mandatory_kw:
        if kw in n: return 1
    for kw in prohib_kw:
        if kw in n: return 0
    return 3


# ------------------------------------------------- download both datasets
def dl(slug, path):
    if path.exists() and (any(path.rglob("*.jpg")) or any(path.rglob("*.png")) or any(path.rglob("*.ppm"))):
        print("already present:", path); return
    path.mkdir(parents=True, exist_ok=True)
    print("downloading", slug, "->", path)
    os.system(f"kaggle datasets download -d {slug} -p {path} --unzip")

dl(GTSDB_SLUG, DL_G)
dl(INDIAN_SLUG, DL_I)


# ------------------------------------------------------ build merged dataset
def find_gtsdb_voc():
    for p in DL_G.rglob("*"):
        if p.is_dir() and p.name == "Annotations" and any(p.glob("*.xml")):
            jpg = p.parent / "JPEGImages"
            if not jpg.exists():
                cand = [q for q in p.parent.rglob("*") if q.is_dir()
                        and (any(q.glob("*.jpg")) or any(q.glob("*.ppm")))]
                jpg = cand[0] if cand else p.parent
            return p, jpg
    return None, None

def find_indian_yolo():
    for y in DL_I.rglob("data.yaml"):
        base = y.parent
        img_dirs = [q for q in base.rglob("*") if q.is_dir() and q.name == "images"]
        lbl_dirs = [q for q in base.rglob("*") if q.is_dir() and q.name == "labels"]
        return (img_dirs[0].parent if img_dirs else base), base, y
    return None, None, None

def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    cx=(xmin+xmax)/2/w; cy=(ymin+ymax)/2/h; bw=(xmax-xmin)/w; bh=(ymax-ymin)/h
    cl=lambda v: max(0.0, min(1.0, v))
    return cl(cx),cl(cy),cl(bw),cl(bh)

def build():
    for split in ("train","val"):
        (DS/split/"images").mkdir(parents=True, exist_ok=True)
        (DS/split/"labels").mkdir(parents=True, exist_ok=True)
    counts={}
    import cv2

    # German
    ann, jpg = find_gtsdb_voc()
    print("GTSDB:", ann, jpg)
    rng = random.Random(SEED)
    if ann:
        for xml in sorted(ann.glob("*.xml")):
            r = ET.parse(xml).getroot(); size=r.find("size")
            w=int(size.find("width").text); h=int(size.find("height").text)
            stem=Path(r.find("filename").text).stem; lines=[]
            for obj in r.findall("object"):
                nm=obj.find("name").text.strip(); fid=FINE_INDEX.get(nm)
                if fid is None and nm.isdigit() and int(nm)<43: fid=int(nm)
                if fid is None: continue
                cls=gtsdb_super(fid); b=obj.find("bndbox")
                cx,cy,bw,bh=voc_to_yolo(float(b.find("xmin").text),float(b.find("ymin").text),
                                        float(b.find("xmax").text),float(b.find("ymax").text),w,h)
                if bw>0 and bh>0:
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    counts[cls]=counts.get(cls,0)+1
            if not lines: continue
            src=jpg/f"{stem}.jpg"
            if not src.exists():
                alt=jpg/f"{stem}.ppm"
                if alt.exists(): src=alt
                else:
                    hits=list(jpg.rglob(f"{stem}.*"))
                    if not hits: continue
                    src=hits[0]
            sp="val" if rng.random()<VAL_FRAC else "train"
            out=DS/sp/"images"/f"de_{stem}.jpg"
            if src.suffix.lower()==".ppm":
                cv2.imwrite(str(out), cv2.imread(str(src)))
            else:
                shutil.copy2(src, out)
            (DS/sp/"labels"/f"de_{stem}.txt").write_text("\n".join(lines))

    # Indian
    import yaml as _yaml
    img_root, lbl_root, yaml_path = find_indian_yolo()
    print("Indian:", img_root, yaml_path)
    if yaml_path:
        cfg=_yaml.safe_load(Path(yaml_path).read_text()); raw=cfg.get("names")
        names=[raw[k] for k in sorted(raw,key=lambda x:int(x))] if isinstance(raw,dict) else list(raw)
        remap={i:indian_super(nm) for i,nm in enumerate(names)}
        print("\n--- Indian class -> super ---")
        for i,nm in enumerate(names): print(f"  {i:3d} {nm:36s} -> {SUPERCLASS_NAMES[remap[i]]}")
        img_index={}
        for ext in ("*.jpg","*.jpeg","*.png"):
            for p in Path(img_root).rglob(ext): img_index.setdefault(p.stem,p)
        label_by_stem={}
        for lp in Path(lbl_root).rglob("*.txt"): label_by_stem.setdefault(lp.stem, lp)
        for stem,lp in label_by_stem.items():
            img=img_index.get(stem)
            if img is None: continue
            out=[]
            for line in lp.read_text().splitlines():
                line=line.strip()
                if not line: continue
                parts=line.split()
                try: cid=int(float(parts[0]))
                except ValueError: continue
                if cid not in remap: continue
                sup=remap[cid]; out.append(" ".join([str(sup)]+parts[1:]))
                counts[sup]=counts.get(sup,0)+1
            if not out: continue
            low=str(lp).lower()
            sp="val" if ("val" in low or "test" in low or rng.random()<VAL_FRAC) else "train"
            shutil.copy2(img, DS/sp/"images"/f"in_{stem}{img.suffix}")
            (DS/sp/"labels"/f"in_{stem}.txt").write_text("\n".join(out))

    (DS/"dataset.yaml").write_text(
        f"path: {DS}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(SUPERCLASS_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i,n in enumerate(SUPERCLASS_NAMES)))
    n_tr=len(list((DS/"train"/"images").glob("*"))); n_va=len(list((DS/"val"/"images").glob("*")))
    print(f"\n{'='*60}\nMERGED train={n_tr} val={n_va}")
    print("boxes/super: "+", ".join(f"{SUPERCLASS_NAMES[k]}={counts.get(k,0)}" for k in range(4)))
    print("="*60)
    if n_tr==0 or n_va==0: raise SystemExit("empty split - datasets not found")

print("Building merged sign dataset...")
build()

# ------------------------------------------------------------------ train
import torch
from ultralytics import YOLO

n_gpu = torch.cuda.device_count()
device = 0 if n_gpu >= 1 else "cpu"
print(f"GPUs: {n_gpu} -> device={device}, batch={BATCH}")

# CONTINUE from the local epoch-20 weights (transfers learned progress).
if CKPT_UPLOAD.exists():
    print(f"continuing from uploaded checkpoint {CKPT_UPLOAD} (+{EPOCHS_MORE} epochs)")
    model = YOLO(str(CKPT_UPLOAD))
else:
    print("!! sign_last_ep20.pt not uploaded - starting from COCO yolov8n instead")
    model = YOLO("yolov8n.pt")

# Crash-proof: sync best.pt to Drive after EVERY epoch, not just at the end.
# A Colab cutoff mid-run then costs at most one epoch, never the whole run.
def _sync_best_to_drive(trainer):
    try:
        b = getattr(trainer, "best", None)
        if b and Path(b).exists() and Path("/content/drive/MyDrive").exists():
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b, DRIVE_OUT / OUT_NAME)
    except Exception as e:
        print(f"  (per-epoch Drive sync skipped: {e})")

model.add_callback("on_fit_epoch_end", _sync_best_to_drive)

model.train(
    data=str(DS / "dataset.yaml"),
    epochs=EPOCHS_MORE, imgsz=IMGSZ, batch=BATCH,
    device=device, workers=2, patience=15,
    project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True,
    pretrained=True, cache=False, save_period=1,
    mosaic=1.0, scale=0.5,
    fliplr=0.0, flipud=0.0,   # NEVER mirror a sign
    hsv_v=0.4, degrees=5.0,
)

best = WORK / "runs" / RUN_NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, WORK / OUT_NAME)
    print(f"\nDONE. Download /content/{OUT_NAME}")
    try:
        if Path("/content/drive/MyDrive").exists():
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, DRIVE_OUT / OUT_NAME)
            print(f"  also saved to {DRIVE_OUT / OUT_NAME}")
    except Exception as e:
        print(f"  (Drive copy skipped: {e})")
else:
    print("\nWARNING: best.pt not found - check logs.")
