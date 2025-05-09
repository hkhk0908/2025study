import os
import csv

img_folder = 'eye_crops'
csv_path = 'labels.csv'

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'label'])

    for img in sorted(os.listdir(img_folder)):
        if 'open_blink' in img:
            label = 'open'
        elif 'closed' in img:
            label = 'closed'
        else:
            label = 'blink'  # 기본값
        writer.writerow([img, label])

print("자동 라벨링 완료!")
