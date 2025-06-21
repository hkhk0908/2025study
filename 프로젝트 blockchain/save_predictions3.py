from datetime import datetime

def save_to_txt(score, filepath='predictions.txt'):
    # 현재 시각과 점수를 파일에 기록
    ts = datetime.now().isoformat(sep=' ', timespec='milliseconds')
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{ts}\t{score:.4f}\n")
