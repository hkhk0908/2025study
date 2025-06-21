import sounddevice as sd                   # 실시간 오디오 입력용 라이브러리
import numpy as np                         # 수치 연산용 라이브러리
from scipy.fft import fft                  # FFT(고속 푸리에 변환) 함수
from sklearn.ensemble import RandomForestClassifier  # 머신러닝 분류기
import joblib                              # 모델 저장/불러오기용
import csv                                 # 로그 파일 저장용
from datetime import datetime              # 현재 시간 기록용
import smtplib                             # 이메일 전송용

# 1. FFT로 오디오 특징 추출하는 함수
def extract_fft_feature(audio):
    fft_result = np.abs(fft(audio))        # 음성 신호를 FFT 후 절댓값으로 에너지 추출
    return fft_result[:512]                # 0~8kHz만 사용 (총 512차원 벡터)

# 2. 머신러닝 모델로 비명 여부 판단
def detect_scream(audio, model):
    feature = extract_fft_feature(audio)   # 오디오 → 주파수 벡터 추출
    feature = feature.reshape(1, -1)       # (1, 512) 형태로 변형 (ML 입력용)
    result = model.predict(feature)        # 예측값: 0(일반), 1(비명)
    return result[0] == 1

# 3. 비명 감지 시 이메일 알림 전송
def send_email():
    server = smtplib.SMTP('smtp.gmail.com', 587)         # Gmail SMTP 서버
    server.starttls()                                    # TLS 보안 연결 시작
    server.login('your_email@gmail.com', 'your_app_password')  # 앱 비밀번호 사용
    msg = 'Subject: [비명 감지]\n\nSoundAI 시스템이 비명을 감지했습니다.'
    server.sendmail('your_email@gmail.com', 'target_email@gmail.com', msg)
    server.quit()

# 4. 감지된 시간과 내용 로그 파일에 저장
def log_detection():
    with open("scream_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), "비명 감지"])

# 5. 마이크에서 들어오는 오디오 스트림을 실시간 분석하는 콜백 함수
def audio_callback(indata, frames, time, status):
    audio = indata[:, 0]                                 # 1채널 마이크 데이터만 추출
    if detect_scream(audio, model):                      # 비명 감지 여부 판단
        print("🔔 비명 감지됨!")
        send_email()                                     # 이메일 전송
        log_detection()                                  # 로그 저장

# 6. 학습된 머신러닝 모델 로드 + 스트리밍 시작
model = joblib.load("scream_rf_model.pkl")               # RandomForest 모델 불러오기

print("🔊 비명 감지 시스템 시작 (CTRL+C로 종료)")
with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
    while True:
        sd.sleep(1000)                                   # 무한 루프 유지
