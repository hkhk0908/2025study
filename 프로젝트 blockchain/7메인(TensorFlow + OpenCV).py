import os, cv2, threading
from multiprocessing import Queue, Process
from inference_worker1 import inference_worker
from overlay_score2 import overlay_score
from save_predictions3 import save_to_txt
from waveform_plot4 import plot_waveform
from onair_gauge5 import show_onair
from radial_chart6 import show_radial

MODEL, ROOT, SEQ = 'blink_model.keras', 'eye_crops', 50

def main():
    q = Queue(200)
    Process(target=inference_worker, args=(MODEL, ROOT, SEQ, q), daemon=True).start()
    scores = []
    # for fn in (plot_waveform, show_onair, show_radial):
    #     threading.Thread(target=fn, args=(lambda s=scores: s,), daemon=True).start()
    while True:
        msg = q.get()
        if msg is None: break
        fname, score = msg
        scores.append(score)
        img = cv2.imread(os.path.join(ROOT, fname))
        cv2.imshow('Inference', overlay_score(img, score))
        save_to_txt(score)
        if cv2.waitKey(1) == 27: break
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
