"""Evaluate sign classifier on the OFFICIAL GTSRB test set (completely unseen data).

The official test set is from different recording sessions than training.
This is the real accuracy measure — no data leakage possible.
"""

import sys
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from sign_classifier import SignCNN
from utils import PROJECT_ROOT, get_class_names


def evaluate_official_test(model_path, data_dir):
    """Evaluate on GTSRB official test set (12,630 unseen images)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = SignCNN(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")

    # Load ground truth
    gt_file = Path(data_dir) / "GT-final_test.csv"
    test_img_dir = Path(data_dir) / "GTSRB" / "Final_Test" / "Images"

    if not gt_file.exists():
        print(f"ERROR: GT file not found: {gt_file}")
        return
    if not test_img_dir.exists():
        print(f"ERROR: Test images not found: {test_img_dir}")
        return

    # Parse GT CSV
    gt_data = []
    with open(gt_file, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            gt_data.append((row['Filename'], int(row['ClassId'])))

    print(f"Official test set: {len(gt_data)} images (completely unseen)")
    print("Evaluating...")

    # Evaluate
    correct = 0
    total = 0
    class_correct = [0] * 43
    class_total = [0] * 43

    IMG_SIZE = 32
    batch_size = 256
    images_batch = []
    labels_batch = []

    for i, (filename, class_id) in enumerate(gt_data):
        img_path = test_img_dir / filename
        if not img_path.exists():
            continue

        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        images_batch.append(img_arr.transpose(2, 0, 1))  # HWC -> CHW
        labels_batch.append(class_id)

        # Process in batches for speed
        if len(images_batch) == batch_size or i == len(gt_data) - 1:
            X = torch.from_numpy(np.array(images_batch)).to(device)
            y = torch.tensor(labels_batch).to(device)

            with torch.no_grad():
                outputs = model(X)
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(y).cpu().numpy()

            for j, (pred, label, is_correct) in enumerate(
                    zip(predicted.cpu(), labels_batch, batch_correct)):
                total += 1
                class_total[label] += 1
                if is_correct:
                    correct += 1
                    class_correct[label] += 1

            images_batch = []
            labels_batch = []

    # Results
    accuracy = correct / total
    print(f"\n{'='*60}")
    print(f"OFFICIAL TEST SET RESULTS (Unseen Data)")
    print(f"{'='*60}")
    print(f"Total images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # Per-class accuracy (worst 5)
    class_names = get_class_names("german")
    class_acc = []
    for i in range(43):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            class_acc.append((i, acc, class_total[i]))

    class_acc.sort(key=lambda x: x[1])
    print(f"\nWorst 5 classes:")
    for cls_id, acc, count in class_acc[:5]:
        print(f"  Class {cls_id:2d} ({class_names[cls_id][:30]:30s}): {acc:.2%} ({count} samples)")

    print(f"\nBest 5 classes:")
    for cls_id, acc, count in class_acc[-5:]:
        print(f"  Class {cls_id:2d} ({class_names[cls_id][:30]:30s}): {acc:.2%} ({count} samples)")

    return accuracy


if __name__ == "__main__":
    model_path = PROJECT_ROOT / "models" / "sign_classifier.pth"
    data_dir = PROJECT_ROOT / "data" / "gtsrb"
    evaluate_official_test(str(model_path), str(data_dir))
