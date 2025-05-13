import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEQ_LEN = 50
X = np.load('X.npy')
y = np.load('y.npy')

print("X.shape:", X.shape)
print("메모리 사용량 (GB):", X.nbytes / (1024**3))

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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_list = []
splits = [(0,500),(500,1000),(1000,1500),(1500,len(X))]
for start,end in splits:
    h = model.fit(X[start:end], y[start:end], epochs=5, batch_size=2)
    history_list.append(h)

model.save('blink_model.keras')
print("5) 모델 저장 완료: blink_model.keras")

# 시각화
acc = sum([h.history['accuracy'] for h in history_list], [])
loss = sum([h.history['loss'] for h in history_list], [])

plt.figure()
plt.plot(acc, label='accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig('accuracy.png')
plt.show()

plt.figure()
plt.plot(loss, label='loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('loss.png')
plt.show()

# 로그 파일 저장
with open("train_log.txt", "w", encoding="utf-8") as f:
    for i, h in enumerate(history_list, 1):
        f.write(f"[history{i} accuracy] {h.history['accuracy']}\n")
        f.write(f"[history{i} loss]     {h.history['loss']}\n")

print("5) 시각화 파일 저장 완료: accuracy.png, loss.png, 로그: train_log.txt")

