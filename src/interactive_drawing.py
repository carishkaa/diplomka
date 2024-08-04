import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import Tk, Button, Label, StringVar, OptionMenu


class DraggableRectangle:
    def __init__(self, rect, ax):
        self.rect = rect
        self.press = None
        self.background = None
        self.ax = ax

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return

        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return

        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        self.rect.set_x(round(x0 + dx))
        self.rect.set_y(round(y0 + dy))

        self.ax.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.ax.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


class InteractivePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Click to add 1x1 rectangles or drag to move them')
        self.rectangles = []  # List to store the rectangles
        self.draggable_rectangles = []

        # Initialize with 10 predefined rectangles
        machines = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), 
                             (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
        for pos in machines:
            self.add_rectangle(pos, facecolor='gray')
        
        for enter in range(1):
            self.add_rectangle((0, 0), facecolor='green')
        for exit in range(1):
            self.add_rectangle((11, 11), facecolor='red')
        
        # robotic_line TODO if you click on (0,0), the new rectangle will be added of color light blue

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def add_rectangle(self, pos, facecolor):
        rect = patches.Rectangle(pos, 1, 1, edgecolor='r', facecolor=facecolor)
        self.ax.add_patch(rect)
        dr = DraggableRectangle(rect, self.ax)
        dr.connect()
        self.rectangles.append(pos)
        self.draggable_rectangles.append(dr)

    def onclick(self, event):
        # Check if the click is inside the axes
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            if x != 0 or y != 0:
                return
            print([x == rect[0] and y == rect[1] for rect in self.rectangles])
            if not any([x == rect[0] and y == rect[1] for rect in self.rectangles]):
                self.add_rectangle((x, y), facecolor='lightblue')
                # self.fig.canvas.draw()
                print(f'Rectangle added at: ({x}, {y})')
                print(f'Current list of rectangles: {self.rectangles}')

    def show(self):
        self.ax.set_xlim(0, 12)
        self.ax.set_ylim(0, 12)
        plt.show()

# Usage
interactive_plot = InteractivePlot()
interactive_plot.show()

# The list of rectangle coordinates can be accessed via interactive_plot.rectangles
