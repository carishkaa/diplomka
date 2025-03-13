import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DraggableRectangle:
    def __init__(self, rect, ax, callback):
        self.rect = rect
        self.press = None
        self.background = None
        self.ax = ax
        self.callback = callback  # Callback to update coordinates in the parent class

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

        new_x = round(x0 + dx)
        new_y = round(y0 + dy)

        # if already exist, do not allow to move
        if (new_x, new_y) in [(round(rect.get_x()), round(rect.get_y())) for rect in self.ax.patches]:
            return

        self.rect.set_x(new_x)
        self.rect.set_y(new_y)

        self.ax.figure.canvas.draw()

        # Update the coordinates in the parent class
        self.callback(self.rect)

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.ax.figure.canvas.draw()

        # Update the coordinates in the parent class
        self.callback(self.rect)

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
        self.color = 'gray'  # Set the fixed color to gray

        # Initialize with 40 predefined rectangles
        initial_positions = [(i, j) for i in range(1, 11) for j in range(1, 11)][:40]
        for pos in initial_positions:
            self.add_rectangle(pos, facecolor='gray')
        
        for enter in range(1):
            self.add_rectangle((0, 0), facecolor='green')
        for exitt in range(1):
            self.add_rectangle((11, 11), facecolor='red')

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def add_rectangle(self, pos, facecolor):
        rect = patches.Rectangle(pos, 1, 1, edgecolor=self.color, facecolor=facecolor)
        self.ax.add_patch(rect)
        dr = DraggableRectangle(rect, self.ax, self.update_coordinates)
        dr.connect()
        self.rectangles.append((pos, rect))
        self.draggable_rectangles.append(dr)

# TODO COPY this part to origin method (update coords)
    def update_coordinates(self, rect):
        for i, (pos, r) in enumerate(self.rectangles):
            if r == rect:
                self.rectangles[i] = ((round(rect.get_x()), round(rect.get_y())), rect)
                break
        print(f'Updated list of rectangles: {[pos for pos, _ in self.rectangles]}')

    def onclick(self, event):
        # Check if the click is inside the axes
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            if x != 0 or y != 0:
                return
            if not any((x, y) == (rx, ry) for (rx, ry), _ in self.rectangles):  # Avoid duplicates
                self.add_rectangle((x, y), facecolor='lightblue')
                self.fig.canvas.draw()
                print(f'Rectangle added at: ({x}, {y}) with color {self.color}')
                print(f'Current list of rectangles: {[pos for pos, _ in self.rectangles]}')

    def show(self):
        self.ax.set_xlim(0, 12)
        self.ax.set_ylim(0, 12)
        plt.show()

# Usage
interactive_plot = InteractivePlot()
interactive_plot.show()

# The list of rectangle coordinates can be accessed via interactive_plot.rectangles
