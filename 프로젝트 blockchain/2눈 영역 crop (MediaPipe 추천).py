import cv2
import os
import mediapipe as mp

frame_folder = 'frames'
save_folder  = 'eye_crops'
os.makedirs(save_folder, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
success = fail = 0

for fn in sorted(os.listdir(frame_folder)):
    img = cv2.imread(os.path.join(frame_folder, fn))
    if img is None:
        fail += 1
        continue

    h, w = img.shape[:2]
    res = mp_face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        fail += 1
        continue

    success += 1
    l  = res.multi_face_landmarks[0]
    x1 = int(l.landmark[33].x * w)
    x2 = int(l.landmark[133].x * w)
    y  = int(l.landmark[33].y * h)
    crop = img[y-20:y+20, x1:x2]

    if crop.size == 0:
        fail += 1
        continue

    crop = cv2.resize(crop, (64, 64))
    cv2.imwrite(os.path.join(save_folder, fn), crop)

print(f"2) 눈 crop 완료 — 성공: {success}, 실패: {fail}")

