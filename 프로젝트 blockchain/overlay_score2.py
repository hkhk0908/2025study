import cv2

def overlay_score(frame, score, threshold=0.5):
    """프레임에 점수를 표시하고, 기준점 넘으면 빨간 테두리 + ON AIR 문구 추가"""
    cv2.putText(frame, f"SCORE: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if score > threshold:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (w-2, h-2), (0, 0, 255), 4)
        cv2.putText(frame, "ON AIR", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
    return frame



