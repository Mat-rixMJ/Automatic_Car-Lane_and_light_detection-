"""
Train a YOLOv8n model that detects German traffic signs.

Strategy: Create synthetic detection dataset by pasting GTSRB sign crops
onto road backgrounds (BDDA frames). Gives us bbox annotations for free.

4 super-classes for detection:
  0: prohibitory (speed limits, no entry, no passing — red circle)
  1: mandatory (go straight, turn left — blue circle)
  2: danger (warning signs — red triangle)
  3: other (priority, yield, stop, etc.)

After detection, the existing GTSRB classifier (96.81%) identifies the exact sign.
"""

import sys
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR, ensure_dirs

DATA_DIR = PROJECT_ROOT / "data" / "sign_det_yolo"
GTSRB_DIR = PROJECT_ROOT / "data" / "gtsrb" / "Train"
BDDA_DIR = PROJECT_ROOT / "BDDA" / "training" / "camera_videos"


def get_superclass(cls_id):
    """Map 43 GTSRB classes to 4 super-classes."""
    if cls_id in range(0, 10) or cls_id in {10, 15, 16, 17, 32, 41, 42}:
        return 0  # prohibitory
    elif cls_id in range(33, 41):
        return 1  # mandatory
    elif cls_id in range(11, 32):
        return 2  # danger
    else:
        return 3  # other


def extract_backgrounds(n=500):
    """Extract random frames from BDDA videos as backgrounds."""
    bg_dir = DATA_DIR / "backgrounds"
    bg_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(bg_dir.glob("*.jpg"))
    if len(existing) >= n:
        print(f"  Backgrounds: {len(existing)} already extracted")
        return existing
    
    videos = list(BDDA_DIR.glob("*.mp4"))
    if not videos:
        print("  ERROR: No BDDA videos found!")
        return []
    
    random.shuffle(videos)
    count = 0
    
    for vid_path in videos:
        if count >= n:
            break
        cap = cv2.VideoCapture(str(vid_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            continue
        
        # Take 2-3 frames from each video
        for _ in range(min(3, n - count)):
            frame_idx = random.randint(0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                out_path = bg_dir / f"bg_{count:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                count += 1
        cap.release()
    
    print(f"  Extracted {count} background frames")
    return list(bg_dir.glob("*.jpg"))


def load_sign_crops():
    """Load GTSRB sign crops grouped by super-class."""
    signs = {0: [], 1: [], 2: [], 3: []}
    
    if not GTSRB_DIR.exists():
        print(f"  ERROR: GTSRB not found at {GTSRB_DIR}")
        return signs
    
    for class_dir in sorted(GTSRB_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_id = int(class_dir.name)
        superclass = get_superclass(cls_id)
        
        # Take up to 50 images per class (enough variety)
        imgs = list(class_dir.glob("*.ppm")) + list(class_dir.glob("*.png"))
        random.shuffle(imgs)
        for img_path in imgs[:50]:
            img = cv2.imread(str(img_path))
            if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
                signs[superclass].append(img)
    
    for k, v in signs.items():
        print(f"    Class {k}: {len(v)} crops")
    return signs


def create_synthetic_dataset(backgrounds, signs, n_train=2000, n_val=400):
    """Paste sign crops onto backgrounds to create detection dataset."""
    
    for split in ["train", "val"]:
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def generate_image(bg_path, idx, split):
        bg = cv2.imread(str(bg_path))
        if bg is None:
            return False
        h, w = bg.shape[:2]
        
        labels = []
        n_signs = random.randint(1, 3)  # 1-3 signs per image
        
        placed = []  # Track placed positions to avoid overlap
        
        for _ in range(n_signs):
            # Pick random super-class
            cls = random.choice([0, 1, 2, 3])
            if not signs[cls]:
                continue
            crop = random.choice(signs[cls])
            
            # Random size (signs appear 30-120px in real dashcam footage)
            sign_size = random.randint(30, min(120, h // 4))
            crop_resized = cv2.resize(crop, (sign_size, sign_size))
            
            # Place in upper 70% of image (signs are above road)
            max_y = int(h * 0.7) - sign_size
            max_x = w - sign_size
            if max_y < 0 or max_x < 0:
                continue
            
            # Try a few positions to avoid overlap
            for _ in range(10):
                px = random.randint(0, max_x)
                py = random.randint(0, max(0, max_y))
                
                # Check overlap with existing placements
                overlap = False
                for (ox, oy, os) in placed:
                    if abs(px - ox) < os and abs(py - oy) < os:
                        overlap = True
                        break
                if not overlap:
                    break
            
            if overlap:
                continue
            
            # Paste with slight augmentation
            region = bg[py:py+sign_size, px:px+sign_size]
            if region.shape[:2] != (sign_size, sign_size):
                continue
            
            # Alpha blend for realism (make edges slightly transparent)
            alpha = random.uniform(0.85, 1.0)
            bg[py:py+sign_size, px:px+sign_size] = cv2.addWeighted(
                crop_resized, alpha, region, 1 - alpha, 0
            )
            
            # YOLO format: class cx cy w h (normalized)
            cx = (px + sign_size / 2) / w
            cy = (py + sign_size / 2) / h
            bw = sign_size / w
            bh = sign_size / h
            labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            placed.append((px, py, sign_size))
        
        if not labels:
            return False
        
        # Save
        cv2.imwrite(str(DATA_DIR / split / "images" / f"{idx:05d}.jpg"), bg)
        with open(DATA_DIR / split / "labels" / f"{idx:05d}.txt", 'w') as f:
            f.write("\n".join(labels))
        return True
    
    # Generate train set
    print(f"  Generating {n_train} training images...")
    count = 0
    attempts = 0
    while count < n_train and attempts < n_train * 2:
        bg_path = random.choice(backgrounds)
        if generate_image(bg_path, count, "train"):
            count += 1
        attempts += 1
        if count % 200 == 0 and count > 0:
            print(f"    {count}/{n_train}")
    print(f"    Train: {count} images")
    
    # Generate val set
    print(f"  Generating {n_val} validation images...")
    count = 0
    attempts = 0
    while count < n_val and attempts < n_val * 2:
        bg_path = random.choice(backgrounds)
        if generate_image(bg_path, count, "val"):
            count += 1
        attempts += 1
    print(f"    Val: {count} images")


def create_dataset_yaml():
    """Create YOLO dataset config."""
    yaml_content = f"""path: {DATA_DIR}
train: train/images
val: val/images

nc: 4
names:
  0: prohibitory
  1: mandatory
  2: danger
  3: other
"""
    yaml_path = DATA_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"  Dataset YAML: {yaml_path}")
    return yaml_path


def train():
    """Fine-tune YOLOv8n on German sign detection."""
    from ultralytics import YOLO
    
    yaml_path = DATA_DIR / "dataset.yaml"
    model = YOLO("yolov8n.pt")  # Start from COCO-pretrained
    
    print("\n" + "=" * 60)
    print("TRAINING: YOLOv8n German Sign Detector")
    print("=" * 60)
    print("Classes: prohibitory, mandatory, danger, other")
    print("Base: YOLOv8n (COCO pretrained — vehicles+lights retained)")
    print()
    
    results = model.train(
        data=str(yaml_path),
        epochs=60,
        imgsz=480,
        batch=8,
        device=0,
        workers=2,
        patience=12,
        save=True,
        project=str(PROJECT_ROOT / "runs"),
        name="german_sign_detector",
        exist_ok=True,
        pretrained=True,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        flipud=0.0,  # Don't flip signs vertically
        fliplr=0.5,
    )
    
    # Copy best model
    best_path = PROJECT_ROOT / "runs" / "german_sign_detector" / "weights" / "best.pt"
    target_path = MODELS_DIR / "german_sign_detector.pt"
    if best_path.exists():
        shutil.copy2(str(best_path), str(target_path))
        print(f"\n✓ Model saved: {target_path}")
    else:
        # Try last.pt
        last_path = PROJECT_ROOT / "runs" / "german_sign_detector" / "weights" / "last.pt"
        if last_path.exists():
            shutil.copy2(str(last_path), str(target_path))
            print(f"\n✓ Model saved (last): {target_path}")
    
    return results


if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("German Sign Detector — Synthetic Dataset + YOLOv8n Training")
    print("=" * 60)
    
    # Step 1: Extract backgrounds from BDDA
    print("\n[1/4] Extracting background frames from BDDA...")
    backgrounds = extract_backgrounds(n=500)
    if not backgrounds:
        print("FATAL: No backgrounds. Check BDDA path.")
        sys.exit(1)
    
    # Step 2: Load GTSRB sign crops
    print("\n[2/4] Loading GTSRB sign crops...")
    signs = load_sign_crops()
    total_crops = sum(len(v) for v in signs.values())
    if total_crops == 0:
        print("FATAL: No sign crops found. Check GTSRB path.")
        sys.exit(1)
    print(f"  Total: {total_crops} crops across 4 classes")
    
    # Step 3: Generate synthetic dataset (skip if already exists)
    yaml_path = DATA_DIR / "dataset.yaml"
    if yaml_path.exists() and (DATA_DIR / "train" / "labels.cache").exists():
        print("\n[3/4] Dataset already exists, skipping generation...")
    else:
        print("\n[3/4] Generating synthetic detection dataset...")
        create_synthetic_dataset(backgrounds, signs, n_train=2000, n_val=400)
    create_dataset_yaml()
    
    # Step 4: Train
    print("\n[4/4] Training YOLOv8n...")
    train()
    
    print("\n" + "=" * 60)
    print("DONE! Model: models/german_sign_detector.pt")
    print("Update run_pipeline_fast.py to use this instead of US detector")
    print("=" * 60)
