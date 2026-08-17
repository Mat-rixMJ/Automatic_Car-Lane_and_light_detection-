"""CARLA Demo — 2 min city drive with ADAS overlay.

Car drives on CARLA autopilot through city with traffic.
Our ADAS system overlays detection + brakes when needed.
Spawns other vehicles + pedestrians for realistic traffic.
Records to output/carla_demo.mp4
"""

import carla
import numpy as np
import cv2
import time
import sys
import random

sys.path.insert(0, 'src')

# --- Config ---
DURATION = 120  # 2 minutes
WIDTH, HEIGHT = 1280, 720
FPS_TARGET = 20
OUTPUT = 'output/carla_demo.mp4'

print('=' * 50)
print('CarLaneI - CARLA ADAS Demo (2 min)')
print('=' * 50)

# Connect
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
world = client.get_world()
print(f'Map: {world.get_map().name}')

# Set weather for good visibility
weather = carla.WeatherParameters.ClearNoon
world.set_weather(weather)
print('Weather: Clear Noon')

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# --- Spawn ego vehicle first (before traffic, to avoid collision) ---
ego_bp = random.choice(bp_lib.filter('vehicle.*'))
ego_vehicle = world.spawn_actor(ego_bp, spawn_points[0])
time.sleep(1)  # Let it settle
ego_vehicle.set_autopilot(True)
print(f'Ego vehicle: {ego_bp.id} (autopilot + ADAS overlay)')

# --- Spawn traffic (away from ego) ---
traffic_vehicles = []
vehicle_bps = bp_lib.filter('vehicle.*')
for i in range(20):
    bp = random.choice(vehicle_bps)
    sp = spawn_points[random.randint(5, len(spawn_points)-1)]  # Not near ego
    try:
        v = world.spawn_actor(bp, sp)
        v.set_autopilot(True)
        traffic_vehicles.append(v)
    except:
        pass
print(f'Traffic spawned: {len(traffic_vehicles)} vehicles')

# --- Camera ---
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', str(WIDTH))
cam_bp.set_attribute('image_size_y', str(HEIGHT))
cam_bp.set_attribute('fov', '100')
cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-5))
camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)

# --- YOLOv8 ---
from ultralytics import YOLO
yolo = YOLO('models/yolov8n.pt')
print('YOLOv8n loaded')

# --- Video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT, fourcc, 20, (WIDTH, HEIGHT))
print(f'Recording to: {OUTPUT}')

# --- Frame callback ---
latest_frame = [None]
def on_image(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))
    latest_frame[0] = arr[:, :, :3].copy()

camera.listen(on_image)

# --- Main loop ---
print(f'\nRunning for {DURATION}s... Press Q to stop early.')
print('Live window: "CarLaneI - CARLA ADAS Demo"')

frame_count = 0
start_time = time.time()
cached_boxes = []

while True:
    elapsed = time.time() - start_time
    if elapsed >= DURATION:
        break

    if latest_frame[0] is None:
        time.sleep(0.01)
        continue

    frame = latest_frame[0].copy()
    frame_count += 1

    # --- Detection (every 2nd frame for speed) ---
    braking = False
    brake_reason = ''

    if frame_count % 2 == 0:
        results = yolo(frame, conf=0.25, verbose=False, imgsz=480)
        cached_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = yolo.names[cls]
                area = (x2 - x1) * (y2 - y1)
                cached_boxes.append((x1, y1, x2, y2, cls, conf, name, area))

    # --- Draw detections + ADAS logic ---
    for (x1, y1, x2, y2, cls, conf, name, area) in cached_boxes:
        cx = (x1 + x2) // 2

        if cls == 9:  # Traffic light
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'Light {conf:.0%}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if area > 2000 and 300 < cx < WIDTH - 300:
                braking = True
                brake_reason = 'TRAFFIC LIGHT'

        elif cls in [2, 5, 7]:  # Car, bus, truck
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, f'{name} {conf:.0%}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
            if area > 50000 and 200 < cx < WIDTH - 200:
                braking = True
                brake_reason = 'COLLISION WARNING'

        elif cls == 0:  # Person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'pedestrian', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            if area > 15000 and 250 < cx < WIDTH - 250:
                braking = True
                brake_reason = 'PEDESTRIAN'

    # --- Apply ADAS brake override ---
    if braking:
        control = carla.VehicleControl()
        control.brake = 0.8
        control.throttle = 0.0
        ego_vehicle.apply_control(control)
    # else autopilot keeps driving

    # --- HUD ---
    try:
        vel = ego_vehicle.get_velocity()
        speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
    except RuntimeError:
        print('Vehicle destroyed - stopping')
        break
    fps = frame_count / elapsed if elapsed > 0 else 0

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, 45), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, f'CarLaneI ADAS | {fps:.0f} FPS | {speed:.0f} km/h | {int(elapsed)}/{DURATION}s',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Brake indicator
    if braking:
        cv2.rectangle(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT), (0, 0, 180), -1)
        cv2.putText(frame, f'ADAS BRAKE: {brake_reason}', (WIDTH // 2 - 180, HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    # Write + display
    writer.write(frame)
    cv2.imshow('CarLaneI - CARLA ADAS Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
camera.stop()
cv2.destroyAllWindows()
writer.release()

camera.destroy()
ego_vehicle.destroy()
for v in traffic_vehicles:
    try:
        v.destroy()
    except:
        pass

total_time = time.time() - start_time
print(f'\nDone! {frame_count} frames in {total_time:.0f}s ({frame_count/total_time:.1f} FPS)')
print(f'Saved: {OUTPUT}')
