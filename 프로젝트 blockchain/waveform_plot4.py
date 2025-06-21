import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_waveform(get_scores, maxlen=200, interval=100):
    """get_scores() 호출해 실시간 파형 플롯"""
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_ylim(0, 1)

    def init():
        line.set_data([], [])
        return line,

    def update(_):
        y = get_scores()
        n = len(y)
        ax.set_xlim(max(0, n - maxlen), n)
        line.set_data(range(n), y)
        return line,

    animation.FuncAnimation(fig, update, init_func=init,
                             interval=interval, blit=True)
    plt.show()



