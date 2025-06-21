import cv2
import os
from tqdm import tqdm

video_root = 'videos'
save_path  = 'frames'
os.makedirs(save_path, exist_ok=True)

label_folders = ['open_blink', 'closed']
count = 0

for lbl in label_folders:
    folder = os.path.join(video_root, lbl)
    files  = [f for f in os.listdir(folder) if f.endswith('.mp4')]
    for vf in tqdm(files, desc=f"[{lbl}] 영상 처리"):
        cap = cv2.VideoCapture(os.path.join(folder, vf))
        while True:
            ret, frame = cap.read()
            if not ret: break
            filename = f"{lbl}_{count:05d}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)
            count += 1
            if count % 100 == 0:
                print(f"> 추출된 프레임 수: {count}")
        cap.release()

print("1) 프레임 추출 완료")
