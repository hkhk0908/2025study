import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------------------
# 1. 데이터 로딩 함수
# ------------------------------
def load_mfcc_from_folder(folder_path, label, n_mfcc=40):
    X, y = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            signal, sr = librosa.load(file_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
            mfcc = mfcc.T
            if mfcc.shape[0] >= 40:
                mfcc = mfcc[:40]  # 길이 통일 (40프레임)
                X.append(mfcc)
                y.append(label)
    return X, y

# ------------------------------
# 2. 비명/일반 소리 데이터 로드
# ------------------------------
X_scream, y_scream = load_mfcc_from_folder("dataset/scream", label=1)
X_normal, y_normal = load_mfcc_from_folder("dataset/normal", label=0)  # 일반 소리 폴더

X = np.array(X_scream + X_normal)
y = np.array(y_scream + y_normal)

X = X[..., np.newaxis]  # (샘플 수, 40, 40, 1)

# ------------------------------
# 3. 학습/검증 데이터 분할
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 4. CNN 모델 구성
# ------------------------------
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(40, 40, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ------------------------------
# 5. 모델 학습
# ------------------------------
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=8)

# ------------------------------
# 6. 모델 저장
# ------------------------------
model.save("scream_cnn_model.keras")
print("✅ 모델 저장 완료: scream_cnn_model.keras")
