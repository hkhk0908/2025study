# 전체 코드에 자세한 주석 추가 + model.json 저장 기능 포함

import cv2  # 실시간 영상 처리용
import mediapipe as mp  # 얼굴 랜드마크 탐지용
import numpy as np  # 수치 연산용
import matplotlib.pyplot as plt  # 실시간 그래프 시각화
from scipy.fft import fft  # FFT: Fast Fourier Transform 주파수 변환
from scipy.ndimage import gaussian_filter1d  # FFT 결과 부드럽게
import json  # 결과 저장용
from collections import deque  # 최근 값 기록용 버퍼
import matplotlib

# === 초기 변수 설정 ===
prev_head_vector = None  # 좌우 방향 변화 감지를 위한 이전 벡터
head_movement_acc = 0.0  # 좌우 방향 누적 각도
prev_pitch_vector = None  # 위아래 방향 변화 감지를 위한 이전 벡터
pitch_movement_acc = 0.0  # 상하 방향 누적 각도
head_down_frames = 0  # 고개 숙임 지속 프레임 수
yaw_deque = deque(maxlen=30)  # 최근 좌우 움직임 기록 (1초)
pitch_deque = deque(maxlen=30)  # 최근 상하 움직임 기록 (1초)

matplotlib.use("TkAgg")  # 윈도우 GUI 환경 실시간 그래프 백엔드 설정

# === EAR 값 이동 평균 스무딩 함수 ===
def smooth_ear(ear_list, window_size=9):
    smoothed = []
    for i in range(len(ear_list)):
        start = max(0, i - window_size + 1)
        window = ear_list[start:i+1]
        smoothed.append(np.mean(window))
    return smoothed

# === 기본 설정 ===
baseline_ear = None  # 기준 EAR
frame_count = 0  # 프레임 수 세기용
fps = 30  # 초당 프레임 수

# === 데이터 저장 구조 ===
ear_history = []  # 전체 EAR 기록
ear_queue = deque(maxlen=150)  # 실시간 EAR 버퍼 (5초 분량)

# === MediaPipe 얼굴 메시 초기화 ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === 눈 랜드마크 인덱스 (왼쪽 / 오른쪽) ===
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

# === EAR 계산 함수 ===
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # 수직 거리 1
    B = np.linalg.norm(eye[2] - eye[4])  # 수직 거리 2
    C = np.linalg.norm(eye[0] - eye[3])  # 수평 거리
    return (A + B) / (2.0 * C)  # EAR 비율 계산

# === 실시간 그래프 초기 설정 ===
plt.ion()
fig, (ax_ear, ax_fft) = plt.subplots(2, 1, figsize=(10, 6))

line_ear, = ax_ear.plot([], [], color='blue')  # EAR 그래프 라인 객체
ax_ear.set_ylim(0, 0.5)
ax_ear.set_xlim(0, 128)
ax_ear.set_title("Real-time EAR")
ax_ear.set_xlabel("Frame")
ax_ear.set_ylabel("EAR")
ax_ear.grid(True)

bar_fft = ax_fft.bar([], [], width=0.3, color='purple')  # FFT 막대 그래프
ax_fft.set_ylim(0, 10)
ax_fft.set_xlim(0, 10)
ax_fft.set_title("Real-time FFT of EAR (Blink Frequency)")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Amplitude")
ax_fft.grid(True)

# === 웹캠 시작 ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in LEFT_EYE_INDEXES])
            right_eye = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in RIGHT_EYE_INDEXES])

            # 눈 점 찍기
            for (x, y) in np.vstack([left_eye, right_eye]).astype(int):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # EAR 계산 및 기록
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            ear_history.append(ear)
            ear_queue.append(ear)

            frame_count += 1
            if baseline_ear is None and frame_count > 150:
                baseline_ear = np.mean(ear_history)
                print(" 개인 baseline EAR 저장:", baseline_ear)

            # 눈 감김 판단
            if baseline_ear is not None and ear < baseline_ear * 0.7:
                cv2.putText(frame, 'Blink!', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # === 고개 방향 처리 ===
            RIGHT_EAR_IDX = 234
            NOSE_IDX = 1
            LEFT_EAR_IDX = 454
            r = np.array([int(landmarks.landmark[RIGHT_EAR_IDX].x * w), int(landmarks.landmark[RIGHT_EAR_IDX].y * h)])
            n = np.array([int(landmarks.landmark[NOSE_IDX].x * w), int(landmarks.landmark[NOSE_IDX].y * h)])
            l = np.array([int(landmarks.landmark[LEFT_EAR_IDX].x * w), int(landmarks.landmark[LEFT_EAR_IDX].y * h)])

            # 선 시각화
            cv2.line(frame, tuple(r), tuple(n), (0, 255, 0), 2)
            cv2.line(frame, tuple(n), tuple(l), (0, 255, 0), 2)

            # 좌우 움직임
            curr_vector = l - r
            if prev_head_vector is not None:
                dot = np.dot(curr_vector, prev_head_vector)
                norm = np.linalg.norm(curr_vector) * np.linalg.norm(prev_head_vector)
                cos_angle = dot / norm if norm != 0 else 1
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                yaw_deque.append(angle_deg if angle_deg > 10 else 0)
            prev_head_vector = curr_vector

            # 상하 움직임
            ear_center = (r + l) / 2
            curr_pitch_vector = n - ear_center
            if prev_pitch_vector is not None:
                dot_p = np.dot(curr_pitch_vector, prev_pitch_vector)
                norm_p = np.linalg.norm(curr_pitch_vector) * np.linalg.norm(prev_pitch_vector)
                cos_p = dot_p / norm_p if norm_p != 0 else 1
                angle_p_rad = np.arccos(np.clip(cos_p, -1.0, 1.0))
                angle_p_deg = np.degrees(angle_p_rad)
                pitch_deque.append(angle_p_deg if angle_p_deg > 10 else 0)
            prev_pitch_vector = curr_pitch_vector

            # 고개 숙임 여부 확인
            ear_avg_y = (r[1] + l[1]) / 2
            nose_y = n[1]
            if nose_y - ear_avg_y > 20:
                head_down_frames += 1
                cv2.putText(frame, "Head Down", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                head_down_frames = max(0, head_down_frames - 1)

    # === 실시간 그래프 갱신 ===
    line_ear.set_ydata(list(ear_queue))
    line_ear.set_xdata(range(len(ear_queue)))
    ax_ear.set_xlim(0, max(10, len(ear_queue)))

    if len(ear_queue) > 32:
        ear_smooth = smooth_ear(list(ear_queue), window_size=9)
        ear_array = np.array(ear_smooth)
        ear_array -= np.mean(ear_array)

        fft_result = np.abs(fft(ear_array))[:len(ear_array) // 2]
        fft_result = gaussian_filter1d(fft_result, sigma=1)

        freqs = np.fft.fftfreq(len(ear_array), d=1 / fps)[:len(ear_array) // 2]
        mask = freqs < 10

        scaled_fft = fft_result[mask] * 5  # ← 이 줄이 반드시 bar() 위에 있어야 함

        ax_fft.cla()
        ax_fft.bar(freqs[mask], scaled_fft, width=0.2, color='purple')
        ax_fft.set_ylim(0, np.max(scaled_fft) * 1.2)

        peak_freq = freqs[np.argmax(fft_result)]
        blinks_per_min = peak_freq * 60

        # === 점수 계산 ===
        if blinks_per_min < 10 or blinks_per_min > 30:
            score = 0
        elif 10 <= blinks_per_min <= 20:
            score = 100
        else:
            score = 50

        if baseline_ear is not None and np.mean(list(ear_queue)[-30:]) < baseline_ear * 0.5:
            score = 0
        if np.mean(yaw_deque) > 12:
            score = max(0, score - 20)
        if np.mean(pitch_deque) > 12:
            score = max(0, score - 20)
        if head_down_frames > 60:
            score = max(0, score - 30)

        cv2.putText(frame, f'Score: {score}', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        print(f"[주파수: {peak_freq:.2f} Hz] [분당 깜빡임: {blinks_per_min:.1f}회] [좌우 평균: {np.mean(yaw_deque):.1f}°] [상하 평균: {np.mean(pitch_deque):.1f}°] [고개숙임: {head_down_frames}프레임] [점수: {score}]", flush=True)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
    cv2.imshow("Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === 종료 처리 ===
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()

# === model.json 저장 ===
if len(ear_history) > 0:
    ear_smooth = smooth_ear(ear_history)
    ear_array = np.array(ear_smooth)
    ear_array -= np.mean(ear_array)
    fft_result = np.abs(fft(ear_array))[:len(ear_array) // 2]
    freqs = np.fft.fftfreq(len(ear_array), d=1 / fps)[:len(ear_array) // 2]
    peak_freq = freqs[np.argmax(fft_result)]
    blinks_per_min = peak_freq * 60

    model_data = {
        "blink_freq_hz": float(peak_freq),
        "blinks_per_minute": float(blinks_per_min),
        "yaw_avg": round(np.mean(yaw_deque), 1),
        "pitch_avg": round(np.mean(pitch_deque), 1),
        "head_down_frames": head_down_frames,
        "score": score
    }

    with open("model.json", "w") as f:
        json.dump(model_data, f, indent=4)
