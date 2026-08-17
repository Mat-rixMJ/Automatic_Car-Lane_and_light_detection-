"""Analyze the quality of pipeline output video."""
import cv2
import numpy as np

cap = cv2.VideoCapture('output/frankfurt_5min_fast.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Output: {w}x{h}, {total} frames, {total/29:.0f}s')

lanes_px = []
drivable_count = 0
vehicle_count = 0
sign_count = 0
tl_count = 0
sampled = 0

for i in range(0, total, 29):  # Every ~1 second
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    sampled += 1

    # Lanes: pure green pixels
    green = cv2.inRange(frame, np.array([0, 240, 0]), np.array([10, 255, 10]))
    lane_pix = green.sum() // 255
    lanes_px.append(lane_pix)

    # Drivable: green tint in bottom half
    bottom = frame[h//2:, :]
    g_diff = float(bottom[:, :, 1].mean()) - float(bottom[:, :, 0].mean())
    if g_diff > 2:
        drivable_count += 1

    # Vehicles: blue-ish boxes
    blue = cv2.inRange(frame, np.array([200, 80, 0]), np.array([255, 130, 30]))
    if blue.sum() // 255 > 50:
        vehicle_count += 1

    # Signs: yellow [0,255,255] boxes in scene area
    sign_area = frame[40:int(h * 0.7), :]
    yellow = cv2.inRange(sign_area, np.array([0, 248, 248]), np.array([8, 255, 255]))
    if yellow.sum() // 255 > 80:
        sign_count += 1

    # Traffic lights: check HUD top-right for RED/GREEN colored text
    hud = frame[0:35, w - 120:]
    red_px = cv2.inRange(hud, np.array([0, 0, 230]), np.array([60, 60, 255])).sum()
    grn_px = cv2.inRange(hud, np.array([0, 230, 0]), np.array([60, 255, 60])).sum()
    ylw_px = cv2.inRange(hud, np.array([0, 230, 230]), np.array([60, 255, 255])).sum()
    if (red_px + grn_px + ylw_px) > 500:
        tl_count += 1

cap.release()

lane_avg = np.mean(lanes_px)
lane_active = sum(1 for x in lanes_px if x > 20)

print(f'\n{"="*55}')
print(f'QUALITY ANALYSIS — frankfurt_5min_fast.mp4')
print(f'{"="*55}')
print(f'Sampled: {sampled} frames (1 per second)')
print(f'')
print(f'LANE DETECTION:')
print(f'  Frames with lanes: {lane_active}/{sampled} ({lane_active/sampled*100:.0f}%)')
print(f'  Avg lane pixels: {lane_avg:.0f} px/frame')
if lane_avg < 80:
    print(f'  ⚠️  LOW QUALITY — lanes barely visible (384px input too small)')
else:
    print(f'  ✓ OK')

print(f'\nDRIVABLE AREA:')
print(f'  Active: {drivable_count}/{sampled} ({drivable_count/sampled*100:.0f}%)')
if drivable_count / sampled > 0.7:
    print(f'  ✓ GOOD')
else:
    print(f'  ⚠️  LOW — green overlay too subtle')

print(f'\nVEHICLE DETECTION:')
print(f'  Active: {vehicle_count}/{sampled} ({vehicle_count/sampled*100:.0f}%)')
if vehicle_count / sampled > 0.4:
    print(f'  ✓ GOOD')
else:
    print(f'  ⚠️  LOW')

print(f'\nSIGN DETECTION:')
print(f'  Frames with signs: {sign_count}/{sampled} ({sign_count/sampled*100:.1f}%)')
if sign_count < 5:
    print(f'  ⚠️  VERY LOW — signs rarely detected')
    print(f'       Causes: 480p resolution makes signs tiny,')
    print(f'       color filter too strict, or signs not in frame')
else:
    print(f'  ✓ OK')

print(f'\nTRAFFIC LIGHTS:')
print(f'  HUD showing state: {tl_count}/{sampled} ({tl_count/sampled*100:.0f}%)')
if tl_count / sampled < 0.1:
    print(f'  ⚠️  LOW — lights detected but HUD state not rendering visibly')
else:
    print(f'  ✓ OK')

print(f'\n{"="*55}')
print(f'SUMMARY OF ISSUES:')
print(f'{"="*55}')
if lane_avg < 80:
    print(f'1. LANE QUALITY: 384px YOLOP input produces thin/sparse lanes')
    print(f'   FIX: Use 480 or 512 input size (tradeoff: ~20 FPS)')
if sign_count < 5:
    print(f'2. SIGN DETECTION: Too few detections')
    print(f'   FIX: Lower area threshold (800→400), lower conf (0.75→0.6)')
    print(f'        and add blue sign detection (German info signs)')
if drivable_count / sampled < 0.8:
    print(f'3. DRIVABLE AREA: Overlay too subtle to notice visually')
    print(f'   FIX: Increase overlay opacity (0.3→0.4)')
