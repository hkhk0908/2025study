import cv2
import mediapipe as mp
import os

frame_folder = 'frames'
save_folder = 'eye_crops'
os.makedirs(save_folder, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

for file in sorted(os.listdir(frame_folder)):
    img_path = os.path.join(frame_folder, file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        x1 = int(landmarks.landmark[33].x * w)
        x2 = int(landmarks.landmark[133].x * w)
        y = int(landmarks.landmark[33].y * h)
        crop = img[y-20:y+20, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (64, 64))
        cv2.imwrite(os.path.join(save_folder, file), crop)

print("눈 crop 완료")
