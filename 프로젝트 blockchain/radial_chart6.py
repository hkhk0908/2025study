import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def show_radial(get_scores, interval=500):
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    ax.set_ylim(0, 1)
    bar = None

    def upd(_):
        nonlocal bar
        vals = get_scores() or [0]
        val = vals[-1]
        ax.clear()
        ax.set_ylim(0, 1)
        bar = ax.bar(0, val, width=0.5, alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_title('Radial Score')
        return bar

    FuncAnimation(fig, upd, interval=interval)
    plt.show()

