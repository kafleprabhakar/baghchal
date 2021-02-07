import matplotlib.pyplot as plt

class PeriodicPlotter():
    def __init__(self, color='r', x_lim=None, y_lim=None):
        plt.ion()
        self.line, = plt.plot([], [], color + '-')
        self.line.axes.set_xlim(*x_lim)
        self.line.axes.set_ylim(*y_lim)

    def plot(self, x, y):
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        plt.draw()
        plt.pause(0.01)