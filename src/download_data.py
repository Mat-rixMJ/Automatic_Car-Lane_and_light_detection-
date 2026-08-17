"""Download datasets and pretrained models.

Usage:
    python src/download_data.py --gtsrb      # Download GTSRB sign dataset
    python src/download_data.py --models     # Download pretrained YOLOP model
    python src/download_data.py --all        # Download everything
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, MODELS_DIR, ensure_dirs


# --- GTSRB Dataset ---
GTSRB_TRAIN_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
GTSRB_TEST_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
GTSRB_TEST_GT_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

# --- YOLOP Pretrained ---
YOLOP_ONNX_URL = "https://github.com/hustvl/YOLOP/releases/download/v1.0/yolop.onnx"
# Note: if the above doesn't work, you can export ONNX from the PyTorch model:
# python export_onnx.py (from the YOLOP repo)


def download_file(url, dest_path):
    """Download with progress."""
    print(f"Downloading: {url}")
    print(f"  -> {dest_path}")

    def hook(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = min(100, downloaded * 100 / total)
            mb = downloaded / (1024 * 1024)
            print(f"\r  {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=hook)
        print()
    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Try downloading manually from the URL above.")
        return False
    return True


def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


def download_gtsrb():
    """Download and organize GTSRB dataset."""
    gtsrb_dir = DATA_DIR / "gtsrb"
    gtsrb_dir.mkdir(parents=True, exist_ok=True)

    train_dir = gtsrb_dir / "Train"
    if train_dir.exists() and any(train_dir.iterdir()):
        print("GTSRB already downloaded. Skipping.")
        return

    print("=" * 60)
    print("Downloading GTSRB (~300 MB, 50K images, 43 classes)")
    print("=" * 60)

    # Training images
    train_zip = gtsrb_dir / "train.zip"
    if not train_zip.exists():
        download_file(GTSRB_TRAIN_URL, str(train_zip))
    extract_zip(str(train_zip), str(gtsrb_dir))

    # Reorganize folder structure
    extracted = gtsrb_dir / "GTSRB" / "Final_Training" / "Images"
    if extracted.exists():
        train_dir.mkdir(exist_ok=True)
        for class_dir in extracted.iterdir():
            if class_dir.is_dir():
                class_id = str(int(class_dir.name))
                dest = train_dir / class_id
                if not dest.exists():
                    class_dir.rename(dest)

    # Test images
    test_zip = gtsrb_dir / "test.zip"
    if not test_zip.exists():
        download_file(GTSRB_TEST_URL, str(test_zip))
    extract_zip(str(test_zip), str(gtsrb_dir))

    # Test ground truth
    gt_zip = gtsrb_dir / "test_gt.zip"
    if not gt_zip.exists():
        download_file(GTSRB_TEST_GT_URL, str(gt_zip))
    extract_zip(str(gt_zip), str(gtsrb_dir))

    print("\nGTSRB download complete!")


def download_models():
    """Download pretrained model weights."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading pretrained models")
    print("=" * 60)

    # 1. YOLOP ONNX
    yolop_path = MODELS_DIR / "yolop.onnx"
    if not yolop_path.exists():
        print("\n[1/4] YOLOP (lane + drivable area + vehicles)")
        success = download_file(YOLOP_ONNX_URL, str(yolop_path))
        if not success:
            print("  Fallback: use PyTorch Hub (auto-downloads):")
            print("  torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True)")
    else:
        print("\n[1/4] YOLOP: already exists ✓")

    # 2. YOLOv8n (COCO pretrained — detects traffic lights as class 9)
    yolov8_path = MODELS_DIR / "yolov8n.pt"
    if not yolov8_path.exists():
        print("\n[2/4] YOLOv8n (COCO — traffic light = class 9)")
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # Auto-downloads to current dir
            # Move to models dir
            default_path = Path("yolov8n.pt")
            if default_path.exists():
                default_path.rename(yolov8_path)
            print(f"  Saved: {yolov8_path}")
        except Exception as e:
            print(f"  Error: {e}")
            print("  Install ultralytics: pip install ultralytics")
    else:
        print("\n[2/4] YOLOv8n: already exists ✓")

    # 3. Traffic Sign Detector (HuggingFace)
    sign_det_path = MODELS_DIR / "sign_detector.pt"
    if not sign_det_path.exists():
        print("\n[3/4] Traffic Sign Detector")
        print("  Download manually from HuggingFace:")
        print("  https://huggingface.co/nezahatkorkmaz/traffic-sign-detection")
        print("  Save as: models/sign_detector.pt")
        print("")
        print("  Or use the color-based fallback (works without this model)")
    else:
        print("\n[3/4] Sign detector: already exists ✓")

    # 4. Traffic Light (LISA-trained)
    tl_path = MODELS_DIR / "traffic_light.pt"
    if not tl_path.exists():
        print("\n[4/4] Traffic Light Detector")
        print("  Download from HuggingFace:")
        print("  https://huggingface.co/dronefreak/lisa-yolov8m")
        print("  Save as: models/traffic_light.pt")
        print("")
        print("  OR: Use YOLOv8n COCO (class 9 = traffic light) — already downloaded above!")
        print("  The pipeline supports both approaches.")
    else:
        print("\n[4/4] Traffic light: already exists ✓")

    # Summary
    print("\n" + "=" * 60)
    print("Model Status:")
    print(f"  YOLOP (onnx):       {'✓' if yolop_path.exists() else '✗'}")
    print(f"  YOLOv8n (coco):     {'✓' if yolov8_path.exists() else '✗'}")
    print(f"  Sign detector:      {'✓' if sign_det_path.exists() else '✗ (optional)'}")
    print(f"  Sign classifier:    {'✓' if (MODELS_DIR / 'sign_classifier.keras').exists() else '✗ (train: python src/sign_classifier.py --train)'}")
    print(f"  Traffic light:      {'✓' if tl_path.exists() else '✗ (using YOLOv8n COCO instead)'}")
    print("=" * 60)
    print("\nMinimum needed to run pipeline: YOLOP + YOLOv8n ✓")
    print("Everything else improves accuracy but isn't required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets and models")
    parser.add_argument("--gtsrb", action="store_true", help="Download GTSRB dataset")
    parser.add_argument("--models", action="store_true", help="Download pretrained models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    ensure_dirs()

    if args.all or args.gtsrb:
        download_gtsrb()
    if args.all or args.models:
        download_models()
    if not (args.all or args.gtsrb or args.models):
        parser.print_help()
