"""Traffic Sign Classifier — CNN trained on GTSRB (PyTorch).

Same architecture as before, but pure PyTorch. No TensorFlow needed.
3 conv blocks + dense head, trained on 32x32 sign crops.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image


class SignCNN(nn.Module):
    """CNN for traffic sign classification. 3 conv blocks + classifier head."""

    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=0), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=0), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=0), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=0), nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_gtsrb_data(data_dir):
    """Load GTSRB from directory: Train/0/*.png, Train/1/*.png, ...

    Returns:
        (X_train, y_train), (X_test, y_test) as numpy arrays.
        X shape: (N, 32, 32, 3) float32 [0-1], y shape: (N,) int
    """
    train_dir = Path(data_dir) / "Train"
    IMG_SIZE = 32
    images, labels = [], []

    print("Loading GTSRB training data...")
    for class_id in range(43):
        class_dir = train_dir / str(class_id)
        if not class_dir.exists():
            continue
        for img_path in list(class_dir.glob("*.png")) + list(class_dir.glob("*.ppm")):
            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            labels.append(class_id)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)

    # Shuffle and split 80/20
    idx = np.random.permutation(len(images))
    images, labels = images[idx], labels[idx]
    split = int(0.8 * len(images))

    X_train, X_test = images[:split], images[split:]
    y_train, y_test = labels[:split], labels[split:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_test, y_test)


def train(data_dir, model_path, epochs=15, batch_size=64, lr=0.001):
    """Train sign classifier on GTSRB."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (X_train, y_train), (X_test, y_test) = load_gtsrb_data(data_dir)

    # Convert to PyTorch tensors (HWC -> CHW)
    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t = torch.from_numpy(X_test.transpose(0, 3, 1, 2)).to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            val_loss = criterion(outputs, y_test_t).item()
            _, predicted = outputs.max(1)
            val_acc = predicted.eq(y_test_t).sum().item() / len(y_test_t)

        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    print(f"Final test accuracy: {val_acc:.4f}")


class SignClassifier:
    """Runtime wrapper for sign classification in the pipeline."""

    def __init__(self, model_path, region="german"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignCNN(num_classes=43).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        sys.path.insert(0, str(Path(__file__).parent))
        from utils import get_class_names
        self.class_names = get_class_names(region)

    def predict_proba(self, roi_bgr):
        """Return the full 43-class probability vector for a BGR crop."""
        import cv2

        img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        return probs.cpu().numpy()

    def classify(self, roi_bgr, min_conf=0.5, min_margin=0.0):
        """Classify a cropped sign image (BGR numpy array).

        GTSRB covers only 43 classes — signs outside it (e.g. U-turn) are
        out-of-distribution and will still be assigned *some* class. A softmax
        margin gate is the cheapest way to reject those: a confident in-set
        prediction separates top-1 from top-2 clearly, an OOD guess does not.

        Returns:
            (class_name, confidence) or None if it fails the gates.
        """
        probs = self.predict_proba(roi_bgr)
        order = np.argsort(probs)[::-1]
        class_id = int(order[0])
        conf = float(probs[class_id])
        margin = conf - float(probs[order[1]])

        if conf < min_conf or margin < min_margin:
            return None
        name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
        return name, conf


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import PROJECT_ROOT, ensure_dirs

    parser = argparse.ArgumentParser(description="Train GTSRB Sign Classifier (PyTorch)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "gtsrb"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "sign_classifier.pth"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()
    if args.train:
        train(args.data_dir, args.model_path, args.epochs, args.batch_size)
    else:
        parser.print_help()
