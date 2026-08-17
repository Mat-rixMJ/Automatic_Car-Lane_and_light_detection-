"""Run the full perception pipeline on a video file.

Crops center panel (for ultra-wide multi-cam footage),
runs YOLOP + YOLOv8 + Sign Classifier, draws ADAS dashboard.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, MODELS_DIR, ensure_dirs
ensure_dirs()


def run_full_pipeline(input_path, output_path, crop_center=True):
    """Run all models on a video file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading models...")

    # 1. YOLOP (lanes + drivable area + vehicles)
    yolop = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True, trust_repo=True)
    yolop.eval()
    if device.type == 'cuda':
        yolop = yolop.cuda()
    print("  YOLOP loaded ✓")

    # 2. YOLOv8n (traffic lights + general objects)
    from ultralytics import YOLO
    yolov8 = YOLO(str(MODELS_DIR / "yolov8n.pt"))
    print("  YOLOv8n loaded ✓")

    # 3. Sign classifier
    from sign_classifier import SignCNN, SignClassifier
    sign_clf = SignClassifier(str(MODELS_DIR / "sign_classifier.pth"), region="german")
    print("  Sign classifier loaded ✓")

    # --- Open video ---
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crop center third if ultra-wide
    if crop_center and orig_w / orig_h > 2.5:
        third = orig_w // 3
        crop_x1, crop_x2 = third, 2 * third
        frame_w = third
    else:
        crop_x1, crop_x2 = 0, orig_w
        frame_w = orig_w
    frame_h = orig_h

    print(f"\nInput: {input_path}")
    print(f"  Original: {orig_w}x{orig_h}, Cropped: {frame_w}x{frame_h}")
    print(f"  Frames: {total_frames}, FPS: {fps:.0f}, Duration: {total_frames/fps:.1f}s")

    # --- Output writer ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, int(fps), (frame_w, frame_h))

    print(f"  Output: {output_path}")
    print(f"\nProcessing...")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, full_frame = cap.read()
        if not ret:
            break

        # Crop center panel
        frame = full_frame[:, crop_x1:crop_x2]
        frame_count += 1
        output = frame.copy()

        # --- YOLOP inference ---
        input_img = cv2.resize(frame, (640, 640))
        input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_rgb.transpose(2, 0, 1)).unsqueeze(0)
        if device.type == 'cuda':
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = yolop(input_tensor)

        # Drivable area mask
        da_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy()
        da_mask_resized = cv2.resize(da_mask.astype(np.uint8), (frame_w, frame_h),
                                     interpolation=cv2.INTER_NEAREST)

        # Lane mask
        ll_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy()
        ll_mask_resized = cv2.resize(ll_mask.astype(np.uint8), (frame_w, frame_h),
                                     interpolation=cv2.INTER_NEAREST)

        # Draw drivable area (semi-transparent green)
        overlay = output.copy()
        overlay[da_mask_resized == 1] = [0, 120, 0]
        output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

        # Draw lane lines (bright green)
        output[ll_mask_resized == 1] = [0, 255, 0]

        # --- YOLOv8 inference (traffic lights, cars, etc.) ---
        yolo_results = yolov8(frame, conf=0.3, verbose=False)
        traffic_light_state = ""

        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name = yolov8.names[cls_id]

                if cls_id == 9:  # Traffic light
                    # Classify state by brightness in ROI thirds
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        h_roi = roi.shape[0]
                        third_h = max(1, h_roi // 3)
                        top_v = roi[:third_h].mean()
                        mid_v = roi[third_h:2*third_h].mean()
                        bot_v = roi[2*third_h:].mean()
                        brightest = max(top_v, mid_v, bot_v)
                        if top_v == brightest:
                            traffic_light_state = "RED"
                            color = (0, 0, 255)
                        elif mid_v == brightest:
                            traffic_light_state = "YELLOW"
                            color = (0, 255, 255)
                        else:
                            traffic_light_state = "GREEN"
                            color = (0, 255, 0)
                    else:
                        color = (0, 255, 255)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output, f"Light: {traffic_light_state}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif cls_id in [2, 5, 7]:  # car, bus, truck
                    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(output, f"{name} {conf:.0%}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        # --- Sign detection (color-based + classifier) ---
        # Only search upper 70% of frame (signs are above road level)
        sign_search_h = int(frame_h * 0.7)
        sign_frame = frame[:sign_search_h, :]
        hsv = cv2.cvtColor(sign_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200 or area > 30000:  # Filter small blobs (taillights)
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if not (0.7 < aspect < 1.4):  # Signs are roughly square
                continue
            roi = sign_frame[y:y+h, x:x+w]
            result = sign_clf.classify(roi)
            if result and result[1] > 0.75:  # Higher threshold to reduce FP
                name, conf = result
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(output, f"{name} {conf:.0%}",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # --- HUD ---
        # Top bar
        hud_overlay = output.copy()
        cv2.rectangle(hud_overlay, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        output = cv2.addWeighted(hud_overlay, 0.6, output, 0.4, 0)

        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(output, f"CarLaneI | Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if traffic_light_state:
            cv2.putText(output, f"[{traffic_light_state}]",
                        (frame_w - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Progress
        if frame_count % 30 == 0:
            pct = frame_count / total_frames * 100
            print(f"  {pct:.0f}% ({frame_count}/{total_frames}) | FPS: {current_fps:.1f}")

        writer.write(output)

    cap.release()
    writer.release()
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"\nDone! {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/video2.mp4")
    parser.add_argument("--output", default="output/pipeline_result.mp4")
    parser.add_argument("--no-crop", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(args.input, args.output, crop_center=not args.no_crop)
