import cv2
import os

# 1. 영상이 저장된 루트 폴더
video_root = 'videos'          # 영상 폴더 위치
save_path = 'frames'           # 프레임 저장 폴더
os.makedirs(save_path, exist_ok=True)

# 2. 라벨 이름 = 폴더 이름 (open_blink / closed 등)
label_folders = ['open_blink', 'closed']
count = 0

# 3. 각 라벨 폴더 안의 영상들을 읽어오기
for label in label_folders:
    folder_path = os.path.join(video_root, label)
    for video_file in os.listdir(folder_path):
        if not video_file.endswith('.mp4'):
            continue

        # ⭐ 이 부분! 영상 열기
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        # 4. 프레임 읽기 및 저장
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            filename = f"{label}_{count:05d}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)
            count += 1
        cap.release()

print("✅ 프레임 추출 완료")
