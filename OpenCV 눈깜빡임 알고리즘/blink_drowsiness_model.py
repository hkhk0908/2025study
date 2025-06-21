# === 라이브러리 불러오기 ===
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# === MediaPipe 초기화 ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# === 웹캠 연결 ===
cap = cv2.VideoCapture(0)

# === 얼굴 랜드마크 인덱스 ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

POSE_LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 78,
    "right_mouth": 308
}

# === 버퍼 및 상태 초기화 ===
ear_buf = deque(maxlen=150)
pitch_buf = deque(maxlen=150)
yaw_buf = deque(maxlen=150)
roll_buf = deque(maxlen=150)
head_down_continuous = 0
score = 100
fps = 30
frame_count = 0

# === FFT 그래프 출력 설정 ===
plt.ion()
fig, ax = plt.subplots(figsize=(14, 10))
line, = ax.plot([], [], lw=3)
ax.set_ylim(0, 30)  # Y축 범위 조정
ax.set_xlim(0, 3)
ax.set_title("Blink Frequency (FFT)", fontsize=18)
ax.set_xlabel("Frequency (Hz)", fontsize=14)
ax.set_ylabel("Amplitude", fontsize=14)

# === 거리 계산 함수 ===
def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# === EAR 계산 함수 ===
def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    hor = get_distance(p1, p4)
    ver = (get_distance(p2, p6) + get_distance(p3, p5)) / 2
    return ver / hor if hor != 0 else 0

# === Euler 각도 계산 (정확도 개선) ===
def get_euler_angles(landmarks, img_shape):
    image_points = np.array([
        landmarks[POSE_LANDMARKS["nose"]][:2],
        landmarks[POSE_LANDMARKS["chin"]][:2],
        landmarks[POSE_LANDMARKS["left_eye_corner"]][:2],
        landmarks[POSE_LANDMARKS["right_eye_corner"]][:2],
        landmarks[POSE_LANDMARKS["left_mouth"]][:2],
        landmarks[POSE_LANDMARKS["right_mouth"]][:2],
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-165.0, 170.0, -135.0),
        (165.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    height, width = img_shape[:2]
    center = (width / 2, height / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(a) for a in euler_angles]

    pitch = math.degrees(math.asin(np.clip(math.sin(pitch), -1.0, 1.0)))
    yaw = math.degrees(math.asin(np.clip(math.sin(yaw), -1.0, 1.0)))
    roll = -math.degrees(math.asin(np.clip(math.sin(roll), -1.0, 1.0)))

    return pitch, yaw, roll

# === FFT 계산 함수 ===
def compute_fft(ear_list):
    arr = np.array(ear_list)
    arr -= np.mean(arr)
    smooth = np.convolve(arr, np.ones(5)/5, mode='same')
    fft_vals = np.abs(np.fft.fft(smooth))[:len(smooth)//2]
    freqs = np.fft.fftfreq(len(smooth), d=1/fps)[:len(smooth)//2]
    return freqs, fft_vals

# === 최대 주파수 추출 ===
def get_peak_frequency(freqs, fft_vals):
    if len(fft_vals) < 2:
        return 0.0
    idx = np.argmax(fft_vals[1:]) + 1
    return freqs[idx]

# === 실시간 처리 루프 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        ih, iw = frame.shape[:2]
        landmarks = [(int(p.x * iw), int(p.y * ih), int(p.z * iw)) for p in face_landmarks.landmark]

        for x, y, _ in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        ear = (calculate_ear(landmarks, LEFT_EYE) + calculate_ear(landmarks, RIGHT_EYE)) / 2
        ear_buf.append(ear)

        pitch, yaw, roll = get_euler_angles(landmarks, frame.shape)
        pitch_buf.append(pitch)
        yaw_buf.append(yaw)
        roll_buf.append(roll)

        head_down_continuous = head_down_continuous + 1 if pitch > 15 else 0

        if len(ear_buf) >= 32:
            freqs, fft_vals = compute_fft(list(ear_buf))
            peak_freq = get_peak_frequency(freqs, fft_vals)
            blinks_per_min = round(peak_freq * 60, 1)
            yaw_std = np.std(yaw_buf)
            yaw_changes = np.sum(np.abs(np.diff(yaw_buf)) > 10)
            closed_frames = sum([1 for e in ear_buf if e < 0.2])

            if blinks_per_min < 12 or closed_frames > 30 or head_down_continuous >= 60 or (yaw_std > 8 and yaw_changes > 15):
                score = max(score - 1, 0)
            else:
                score = min(score + 1, 100)

            cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Blink/min: {blinks_per_min}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

            if frame_count % 5 == 0:
                line.set_xdata(freqs)
                line.set_ydata(fft_vals)
                ax.set_ylim(0, max(10, np.max(fft_vals)*1.2))  # y축 자동 조정
                fig.canvas.draw()
                fig.canvas.flush_events()

    cv2.imshow("Blink + Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
