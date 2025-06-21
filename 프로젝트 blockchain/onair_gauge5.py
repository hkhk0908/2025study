import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def show_onair(get_scores, thr=0.5, interval=500):
    fig, ax = plt.subplots(); ax.axis('off'); ax.set_aspect('equal')
    theta = np.linspace(0, np.pi, 200)
    def upd(_):
        ax.clear(); ax.axis('off'); ax.set_aspect('equal')
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=2)
        s = get_scores()[-1] if get_scores() else 0
        a = np.pi * s
        ax.arrow(0,0,np.cos(a),np.sin(a),width=0.02,head_length=0.05,color='red')
        ax.set_title('ON AIR' if s>thr else 'OFF AIR', color='red' if s>thr else 'green')
    FuncAnimation(fig, upd, interval=interval)
    plt.show()


