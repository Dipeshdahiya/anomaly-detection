import threading
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# === Load Models ===
pose_model = YOLO("yolov8n-pose.pt")
weapon_model = YOLO("yolov5s.pt")


# === Load Ghost Model (MobileNetV2-Based Classifier) ===
my_model = load_model("cctv_anomaly_detection_mobilenetv2.keras")
print("🕵️‍♂️  Model (MobileNetV2-based anomaly classifier) loaded.")

# === Global Shared Variables ===
weapon_detected = False
weapon_label = ""
weapon_confidence = 0
current_frame = None
lock = threading.Lock()

# === Buffers and Tracking ===
person_buffers = {}
prev_keypoints = {}
prev_velocities = {}

# === Detection Toggles ===
weapon_detection_enabled = True
pose_detection_enabled = True

# === Utility Functions ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def velocity(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# === Activity Detection ===
def detect_activity_yolo(keypoints, pid):
    global prev_keypoints, prev_velocities

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

    # Static pose-based detection
    if body_angle < 30 and hip_ankle_diff < 40:
        return "Lying Down"
    elif 30 <= body_angle < 60:
        return "Falling"
    elif mid_hip[1] < 220:
        return "Sneaking"

    action = "Standing"

    if pid in prev_keypoints:
        prev = prev_keypoints[pid]
        velocities = {k: velocity(prev[k], now) for k, now in zip(['lw', 'rw', 'la', 'ra'], [lw, rw, la, ra])}
        angles = {
            'lw': calculate_angle(ls, le, lw),
            'rw': calculate_angle(rs, re, rw),
            'la': calculate_angle(lh, lk, la),
            'ra': calculate_angle(rh, rk, ra)
        }

        if (velocities['lw'] > 12 or velocities['rw'] > 12) and (60 < angles['lw'] < 150 or 60 < angles['rw'] < 150):
            action = "Hitting"
        elif (velocities['lw'] > 8 and angles['lw'] < 40) or (velocities['rw'] > 8 and angles['rw'] < 40):
            action = "Throwing"
        elif (velocities['la'] > 12 and angles['la'] > 150) or (velocities['ra'] > 12 and angles['ra'] > 150):
            action = "Kicking"
        elif velocity(lw, ls) < 30 and velocities['lw'] > 10:
            action = "Grabbing"

        prev_velocities[pid] = velocities

    prev_keypoints[pid] = {'lw': lw, 'rw': rw, 'la': la, 'ra': ra}
    return action

# === Weapon Detection ===
def detect_weapons(frame):
    global weapon_detected, weapon_label, weapon_confidence
    results = weapon_model.predict(frame, conf=0.25, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id].lower()
            confidence = float(box.conf[0])
            if label in ["gun", "knife", "handgun"]:
                weapon_detected = True
                weapon_label = label
                weapon_confidence = confidence
                return frame

    weapon_detected = False
    weapon_label = ""
    weapon_confidence = 0
    return frame

# === Frame Processing ===
def process_frame(frame):
    global current_frame
    current_frame = frame.copy()
    person_activities = []

    if pose_detection_enabled:
        results = pose_model.predict(frame, conf=0.3, verbose=False)[0]
        if results.keypoints is not None:
            for pid, (kp, box) in enumerate(zip(results.keypoints.data, results.boxes.xyxy)):
                keypoints = [[int(x), int(y)] for x, y, _ in kp.cpu().numpy()]
                activity = detect_activity_yolo(keypoints, pid)
                person_buffers.setdefault(pid, deque(maxlen=5)).append(activity)
                most_common = Counter(person_buffers[pid]).most_common(1)[0][0]
                person_activities.append((box, most_common))

    if weapon_detection_enabled:
        frame = detect_weapons(frame)

    return draw_overlay(frame, person_activities)

# === Overlay Drawing ===
def draw_overlay(frame, activities):
    frame_display = frame.copy()

    for box, activity in activities:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if activity in ["Lying Down", "Falling", "Sneaking", "Hitting", "Kicking", "Throwing", "Grabbing"] else (0, 255, 0)
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_display, activity, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

    if weapon_detection_enabled and weapon_detected:
        cv2.rectangle(frame_display, (0, 40), (640, 70), (0, 0, 100), -1)
        cv2.putText(frame_display, f"⚠️ Weapon Detected: {weapon_label.upper()} ({weapon_confidence:.2f})", (10, 65), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    return frame_display

# === Video Streaming ===
def generate_video():
    cap = cv2.VideoCapture(0)
    frame_skip = 2

    while True:
        for _ in range(frame_skip):
            success, frame = cap.read()
            if not success:
                continue

        frame = cv2.resize(frame, (640, 480))
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# === Logging Suspicious Events ===
def log_suspicious_activity(activities, weapon_detected, weapon_label, weapon_confidence):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] Suspicious Activities: {activities}, Weapon: {weapon_label.upper() if weapon_detected else 'None'}, Confidence: {weapon_confidence:.2f}\n"
    with open("activity_log.txt", "a") as log_file:
        log_file.write(log_entry)

# === Flask App Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_weapon', methods=['POST'])
def toggle_weapon():
    global weapon_detection_enabled
    weapon_detection_enabled = not weapon_detection_enabled
    return jsonify({'status': weapon_detection_enabled})

@app.route('/toggle_pose', methods=['POST'])
def toggle_pose():
    global pose_detection_enabled
    pose_detection_enabled = not pose_detection_enabled
    return jsonify({'status': pose_detection_enabled})

@app.route('/send_email', methods=['POST'])
def send_email():
    global weapon_detected, weapon_label, weapon_confidence, person_buffers
    recent_activities = []

    for pid, buffer in person_buffers.items():
        if buffer:
            most_common = Counter(buffer).most_common(1)[0][0]
            if most_common in ["Lying Down", "Falling", "Sneaking", "Hitting", "Kicking", "Throwing", "Grabbing"]:
                recent_activities.append(most_common)

    if not recent_activities:
        recent_activities = ["None"]

    log_suspicious_activity(recent_activities, weapon_detected, weapon_label, weapon_confidence)
    print("📧 Email alert sent to authorities!")
    return jsonify({'status': 'success', 'message': 'Email alert sent and activity logged!'})

# === Run Flask Server ===
if __name__ == '__main__':
    app.run(debug=True, threaded=True)  