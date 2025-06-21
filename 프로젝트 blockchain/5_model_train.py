#!/usr/bin/env python3
# 5_model_train.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, TimeDistributed,
    Conv2D, MaxPooling2D,
    Flatten, LSTM, Dense
)
from tensorflow.keras.utils import to_categorical

# 설정
IMG_FOLDER = 'eye_crops'
LABEL_CSV  = 'labels.csv'
SEQ_LEN    = 50

# 1) 데이터 로드 & 전처리
df    = pd.read_csv(LABEL_CSV)
paths = [os.path.join(IMG_FOLDER, n) for n in df['image']]
labs  = df['label'].map({'open':0,'closed':1,'blink':2}).values

X, y = [], []
for i in range(len(paths) - SEQ_LEN):
    seq = []
    for p in paths[i:i+SEQ_LEN]:
        img = cv2.imread(p)
        img = cv2.resize(img, (64,64))
        seq.append(img/255.0)
    X.append(seq)
    y.append(labs[i+SEQ_LEN-1])

X = np.array(X)                    # (N,50,64,64,3)
y = to_categorical(y, num_classes=3)

print(f"[Data] X.shape={X.shape}, y.shape={y.shape}")

# 2) 모델 정의
model = Sequential([
    Input(shape=(SEQ_LEN,64,64,3)),
    TimeDistributed(Conv2D(32,(3,3),activation='relu')),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Conv2D(64,(3,3),activation='relu')),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3) 학습
history = model.fit(
    X, y,
    epochs=20,
    batch_size=8,
    validation_split=0.1
)

# 4) 저장
model.save('blink_model.keras')
print("▶ 모델 저장 완료: blink_model.keras")

# 5) 시각화 결과 저장
plt.figure(); plt.plot(history.history['accuracy'], label='acc'); plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy'); plt.savefig('accuracy.png')

plt.figure(); plt.plot(history.history['loss'], label='loss'); plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss'); plt.savefig('loss.png')

print("▶ 그래프 저장 완료: accuracy.png, loss.png")
