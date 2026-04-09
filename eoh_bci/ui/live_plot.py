import matplotlib.pyplot as plt
from collections import deque

class LivePlot:
    def __init__(self, max_points=100):
        self.data = deque(maxlen=max_points)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([])

        self.ax.set_ylim(0, 1)  # adjust based on signal range
        self.ax.set_title("EEG Signal (Real-Time)")

    def update(self, value):
        if value is None:
            return

        self.data.append(value)

        self.line.set_xdata(range(len(self.data)))
        self.line.set_ydata(self.data)

        self.ax.set_xlim(0, len(self.data))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()