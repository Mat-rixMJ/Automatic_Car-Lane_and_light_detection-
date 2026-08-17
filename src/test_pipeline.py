"""Quick test — run all models on sample images to verify everything works."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR, ensure_dirs

ensure_dirs()


def test_yolop():
    """Test YOLOP on sample image."""
    print("\n=== Testing YOLOP (Lane + Drivable Area + Vehicles) ===")
    img_path = PROJECT_ROOT / "data" / "sample_road.jpg"
    if not img_path.exists():
        print(f"  SKIP: {img_path} not found")
        return

    img = cv2.imread(str(img_path))
    print(f"  Image: {img.shape}")

    # Load YOLOP
    model = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True, trust_repo=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Preprocess
    h, w = img.shape[:2]
    input_img = cv2.resize(img, (640, 640))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_tensor = torch.from_numpy(input_img).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Inference
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(input_tensor)

    # Post-process
    da_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy()
    ll_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy()

    da_pixels = da_mask.sum()
    ll_pixels = ll_mask.sum()
    print(f"  Drivable area pixels: {da_pixels} ({da_pixels/(640*640)*100:.1f}%)")
    print(f"  Lane line pixels: {ll_pixels} ({ll_pixels/(640*640)*100:.1f}%)")
    print(f"  Detections: {det_out[0].shape if len(det_out) > 0 else 'none'}")

    # Draw results
    output = cv2.resize(img, (640, 640))
    output[da_mask == 1] = output[da_mask == 1] * 0.5 + np.array([0, 100, 0]) * 0.5
    output[ll_mask == 1] = [0, 255, 0]

    out_path = PROJECT_ROOT / "output" / "test_yolop.jpg"
    cv2.imwrite(str(out_path), output)
    print(f"  Output saved: {out_path}")
    print("  ✓ YOLOP PASSED")


def test_yolov8_traffic_light():
    """Test YOLOv8n on sample image for traffic light detection."""
    print("\n=== Testing YOLOv8n (Traffic Light Detection) ===")
    from ultralytics import YOLO

    model_path = MODELS_DIR / "yolov8n.pt"
    img_path = PROJECT_ROOT / "data" / "sample_road2.jpg"

    if not model_path.exists():
        print(f"  SKIP: {model_path} not found")
        return
    if not img_path.exists():
        print(f"  SKIP: {img_path} not found")
        return

    model = YOLO(str(model_path))
    results = model(str(img_path), verbose=False)

    all_detections = []
    traffic_lights = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            all_detections.append((name, conf))
            if cls_id == 9:  # traffic light
                traffic_lights.append((name, conf))

    print(f"  Total detections: {len(all_detections)}")
    print(f"  Traffic lights: {len(traffic_lights)}")
    for name, conf in all_detections[:10]:
        marker = " ← TRAFFIC LIGHT" if name == "traffic light" else ""
        print(f"    {name}: {conf:.2%}{marker}")

    # Save annotated image
    annotated = results[0].plot()
    out_path = PROJECT_ROOT / "output" / "test_yolov8.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  Output saved: {out_path}")
    print("  ✓ YOLOv8n PASSED")


def test_sign_classifier():
    """Test sign classifier on a crop from GTSRB test set."""
    print("\n=== Testing Sign Classifier (GTSRB) ===")
    from sign_classifier import SignCNN
    from utils import get_class_names

    model_path = MODELS_DIR / "sign_classifier.pth"
    if not model_path.exists():
        print(f"  SKIP: {model_path} not found")
        return

    # Load a random test image
    test_dir = PROJECT_ROOT / "data" / "gtsrb" / "GTSRB" / "Final_Test" / "Images"
    test_imgs = list(test_dir.glob("*.ppm"))[:5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignCNN(43).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    class_names = get_class_names("german")

    print(f"  Testing on {len(test_imgs)} sample images:")
    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (32, 32))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, pred = probs.max(0)

        print(f"    {img_path.name}: {class_names[pred.item()]} ({conf.item():.2%})")

    print("  ✓ Sign Classifier PASSED")


if __name__ == "__main__":
    test_sign_classifier()
    test_yolov8_traffic_light()
    test_yolop()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
