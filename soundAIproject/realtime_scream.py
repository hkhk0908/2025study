import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import rfft
import librosa
from tensorflow.keras.models import load_model
import queue
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')  # 한글 깨짐 방지 (Windows 기준)

# ------------------------------
# 설정
# ------------------------------
fs = 16000           # 샘플링 레이트 (Hz)
duration = 1.0       # 처리 단위 시간 (초)
blocksize = int(fs * duration)
model_path = "scream_cnn_model.keras"  # 저장된 모델 경로

# ------------------------------
# 모델 로드
# ------------------------------
model = load_model(model_path)

# ------------------------------
# 오디오 큐 설정
# ------------------------------
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# ------------------------------
# MFCC 추출 함수
# ------------------------------
def extract_mfcc(audio_data, sr=16000, n_mfcc=40):
    audio_data = audio_data.flatten()
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    if mfcc.shape[0] < 40:
        return None
    mfcc = mfcc[:40]
    return mfcc[..., np.newaxis]

# ------------------------------
# FFT 시각화 초기 설정
# ------------------------------
plt.ion()
fig, ax = plt.subplots()
x_fft = np.fft.rfftfreq(blocksize, d=1/fs)
line, = ax.plot(x_fft, np.zeros_like(x_fft))
ax.set_ylim(0, 1000)
ax.set_xlim(0, 4000)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("실시간 FFT + 비명 감지")

# ------------------------------
# 메인 실행 루프
# ------------------------------
with sd.InputStream(channels=1, samplerate=fs, callback=audio_callback, blocksize=blocksize):
    print("🎤 실시간 비명 감지 시작! (Ctrl+C로 종료)\n")

    while True:
        audio_block = q.get()
        audio_np = audio_block[:, 0]

        # 실시간 FFT 계산 및 그래프 업데이트
        fft_result = np.abs(rfft(audio_np))
        line.set_ydata(fft_result)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 모델 입력 전처리 (MFCC)
        mfcc = extract_mfcc(audio_np)
        if mfcc is not None:
            mfcc_input = np.expand_dims(mfcc, axis=0)  # (1, 40, 40, 1)
            pred = model.predict(mfcc_input)[0][0]
            print(f"예측 확률: {pred:.2f}")
            if pred > 0.8:
                print("🚨 비명 감지됨!")
