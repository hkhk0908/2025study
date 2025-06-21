import os
import csv
import pandas as pd

img_folder = 'eye_crops'
csv_path   = 'labels.csv'

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'label'])
    for img in sorted(os.listdir(img_folder)):
        if 'open_blink' in img:
            label = 'open'
        elif 'closed' in img:
            label = 'closed'
        else:
            label = 'blink'
        writer.writerow([img, label])

# 분포 확인
df = pd.read_csv(csv_path)
print("3) 라벨 분포:\n", df['label'].value_counts())
