import os
import shutil

target_folder = 'web_package'
eye_crop_src  = 'eye_crops'
files_to_copy = [
    'labels.csv',
    'X.npy',
    'y.npy',
    'blink_model.keras',
    'train_log.txt',
    'accuracy.png',
    'loss.png'
]

# 폴더 초기화
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
os.makedirs(target_folder)

# 파일 복사
for fn in files_to_copy:
    shutil.copy(fn, target_folder)

# 눈 크롭 이미지 폴더 전체 복사
shutil.copytree(eye_crop_src, os.path.join(target_folder, 'eye_crops'))

print(f"6) 웹 패키지 준비 완료 → ./{target_folder}/")
