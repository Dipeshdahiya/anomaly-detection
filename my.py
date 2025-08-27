import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
from tensorflow.keras.models import load_model

# === Load Models ===
pose_model = YOLO("yolov8n-pose.pt")
weapon_model = YOLO("yolov5su.pt")  # Your custom weapon detection model

# === Load Ghost Model (MobileNetV2-Based Classifier) ===
my_model = load_model("cctv_anomaly_detection_mobilenetv2.keras")
print("🕵️‍♂️  Model (MobileNetV2-based anomaly classifier) loaded.")

# === Shared Variables ===
weapon_detected = False
weapon_label = ""
weapon_confidence = 0
current_frame = None
lock = threading.Lock()

# === Buffers and Trackers ===
person_buffers = {}
prev_keypoints = {}
last_sent = {'activity': None, 'weapon': False}

# === Overlay Toggle Flags ===
show_pose_overlay = True
show_weapon_box = True

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def detect_activity_yolo(keypoints, pid):
    global prev_keypoints

    def get_point(index):
        return [keypoints[index][0], keypoints[index][1]]

    ls, rs = get_point(5), get_point(6)
    le, re = get_point(7), get_point(8)
    lw, rw = get_point(9), get_point(10)
    lh, rh = get_point(11), get_point(12)
    lk, rk = get_point(13), get_point(14)
    la, ra = get_point(15), get_point(16)

    mid_shoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
    mid_hip = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
    hip_ankle_diff = abs(mid_hip[1] - ((la[1] + ra[1]) / 2))
    body_angle = calculate_angle(la, mid_hip, mid_shoulder)

    if body_angle < 30 and hip_ankle_diff < 40:
        return "Lying Down"
    elif 30 <= body_angle < 60:
        return "Falling"
    elif mid_hip[1] < 220:
        return "Sneaking"

    action = "Standing"

    if pid in prev_keypoints:
        prev = prev_keypoints[pid]
        lw_speed = np.linalg.norm(np.array(lw) - np.array(prev['lw']))
        rw_speed = np.linalg.norm(np.array(rw) - np.array(prev['rw']))
        la_speed = np.linalg.norm(np.array(la) - np.array(prev['la']))
        ra_speed = np.linalg.norm(np.array(ra) - np.array(prev['ra']))

        lw_angle = calculate_angle(ls, le, lw)
        rw_angle = calculate_angle(rs, re, rw)
        la_angle = calculate_angle(lh, lk, la)
        ra_angle = calculate_angle(rh, rk, ra)

        if (lw_speed > 12 or rw_speed > 12) and (60 < lw_angle < 150 or 60 < rw_angle < 150):
            action = "Hitting"
        elif (lw_speed > 8 and lw_angle < 40) or (rw_speed > 8 and rw_angle < 40):
            action = "Throwing"
        elif (la_speed > 12 and la_angle > 150) or (ra_speed > 12 and ra_angle > 150):
            action = "Kicking"
        elif np.linalg.norm(np.array(lw) - np.array(ls)) < 30 and lw_speed > 10:
            action = "Grabbing"

    prev_keypoints[pid] = {'lw': lw, 'rw': rw, 'la': la, 'ra': ra}
    return action

def detect_weapons():
    global weapon_detected, current_frame, weapon_label, weapon_confidence, last_sent
    while True:
        time.sleep(0.01)
        with lock:
            if current_frame is None:
                continue
            frame_copy = current_frame.copy()

        results = weapon_model.predict(frame_copy, conf=0.25, verbose=False)
        detected = False
        label, confidence = "", 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = result.names[class_id]
                confidence = float(box.conf[0])
                if label.lower() in ["gun", "knife", "handgun"]:
                    detected = True
                    break

        with lock:
            weapon_detected = detected
            weapon_label = label
            weapon_confidence = confidence

# Start background thread
threading.Thread(target=detect_weapons, daemon=True).start()

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    with lock:
        current_frame = frame.copy()

    results = pose_model.predict(frame, conf=0.3, verbose=False)[0]

    person_activities = []
    if results.keypoints is not None:
        for pid, (kp, box) in enumerate(zip(results.keypoints.data, results.boxes.xyxy)):
            keypoints = kp.cpu().numpy()
            keypoints = [[int(x), int(y)] for x, y, c in keypoints]

            activity = detect_activity_yolo(keypoints, pid)
            person_buffers.setdefault(pid, deque(maxlen=5)).append(activity)
            most_common_activity = Counter(person_buffers[pid]).most_common(1)[0][0]
            person_activities.append((box, most_common_activity))

    # === Drawing Frame ===
    frame_display = frame.copy()

    if show_pose_overlay:
        for box, activity in person_activities:
            x1, y1, x2, y2 = map(int, box)
            suspicious = activity in ["Lying Down", "Falling", "Sneaking", "Hitting", "Kicking", "Throwing", "Grabbing"]
            color = (0, 0, 255) if suspicious else (0, 255, 0)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_display, activity, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

    cv2.rectangle(frame_display, (0, 0), (640, 40), (50, 50, 50), -1)
    if show_weapon_box and weapon_detected:
        cv2.rectangle(frame_display, (0, 40), (640, 70), (0, 0, 100), -1)
        cv2.putText(frame_display, f"⚠️ Weapon Detected: {weapon_label.upper()} ({weapon_confidence:.2f})",
                    (10, 65), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"🚨 Weapon: {weapon_label.upper()} ({weapon_confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    elif show_pose_overlay:
        cv2.putText(frame_display, "Monitoring for suspicious activities...", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("🔍 Multi-Person Suspicious Activity Detection", frame_display)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        show_pose_overlay = not show_pose_overlay
        print(f"🔁 Pose Overlay: {'ON' if show_pose_overlay else 'OFF'}")
    elif key == ord('w'):
        show_weapon_box = not show_weapon_box
        print(f"🔁 Weapon Box: {'ON' if show_weapon_box else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
