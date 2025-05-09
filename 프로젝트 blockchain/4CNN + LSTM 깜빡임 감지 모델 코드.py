import numpy as np
import cv2
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical

img_folder = 'eye_crops'
label_csv = 'labels.csv'
SEQ_LEN = 10

df = pd.read_csv(label_csv)
img_paths = [os.path.join(img_folder, name) for name in df['image']]
labels = df['label'].map({'open': 0, 'closed': 1, 'blink': 2}).values

X, y = [], []
for i in range(len(img_paths) - SEQ_LEN):
    seq_imgs = img_paths[i:i+SEQ_LEN]
    seq_label = labels[i+SEQ_LEN-1]

    imgs = [cv2.imread(p) for p in seq_imgs]
    imgs = [cv2.resize(img, (64, 64)) for img in imgs]
    imgs = [img / 255.0 for img in imgs]

    X.append(imgs)
    y.append(seq_label)

X = np.array(X)
y = to_categorical(y, num_classes=3)

np.save('X.npy', X)
np.save('y.npy', y)
print("시퀀스 저장 완료")

