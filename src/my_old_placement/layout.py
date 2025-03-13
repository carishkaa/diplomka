import numpy as np

def get_layout(type): 
    if type == 'rectangle':
        COORDINATE_X = 4
        COORDINATE_Y = 20
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1

        first_row = [(0, i) for i in range(1,COORDINATE_Y-1)]
        last_column = [(i, COORDINATE_Y-1) for i in range(1,COORDINATE_X-1)]
        last_row = [(COORDINATE_X-1, i) for i in range(1,COORDINATE_Y-1)]
        first_column = [(i, 0) for i in range(1,COORDINATE_X-1)]
        all_posible_positions = first_row + last_column + last_row[::-1] + first_column[::-1]
    elif type == 'square_1_enter_1_exit':
        COORDINATE_X = 13
        COORDINATE_Y = 12
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1

        first_row = [(0, i) for i in range(1,COORDINATE_Y-1)]
        last_column = [(i, COORDINATE_Y-1) for i in range(1,COORDINATE_X-1)]
        last_row = [(COORDINATE_X-1, i) for i in range(1,COORDINATE_Y-1)]
        first_column = [(i, 0) for i in range(1,COORDINATE_X-1)]
        all_posible_positions = first_row + last_column + last_row[::-1] + first_column[::-1]
    elif type == 'doubeline_1_enter_50': # 1 enter, 50 machines
        COORDINATE_X = 26
        COORDINATE_Y = 2
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[0, 0] = 1
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'doubeline_2_enter_50': # 2 inlets, 50 machines
        COORDINATE_X = 26
        COORDINATE_Y = 2
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'doubeline_4_enter_50':
        COORDINATE_X = 27
        COORDINATE_Y = 2
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'doubleline_4_enter_80':
        COORDINATE_X = 42
        COORDINATE_Y = 2
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'doubleline_4_enter_60':
        COORDINATE_X = 32
        COORDINATE_Y = 2
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'kolecko_2_enter_50':
        COORDINATE_X = 14
        COORDINATE_Y = 14
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        # restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1
        # positions as a tape: first row, last column, last row, first column
        first_row = [(0, i) for i in range(0,COORDINATE_Y)]
        last_column = [(i, COORDINATE_Y-1) for i in range(1,COORDINATE_X-1)]
        last_row = [(COORDINATE_X-1, i) for i in range(1,COORDINATE_Y)]
        first_column = [(i, 0) for i in range(1,COORDINATE_X)]
        all_posible_positions = first_row + last_column + last_row[::-1] + first_column[::-1]
    elif type == 'kolecko_4_enter_50':
        COORDINATE_X = 14
        COORDINATE_Y = 15
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        # restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1
        # positions as a tape: first row, last column, last row, first column
        first_row = [(0, i) for i in range(0,COORDINATE_Y)]
        last_column = [(i, COORDINATE_Y-1) for i in range(1,COORDINATE_X-1)]
        last_row = [(COORDINATE_X-1, i) for i in range(1,COORDINATE_Y)]
        first_column = [(i, 0) for i in range(1,COORDINATE_X)]
        all_posible_positions = first_row + last_column + last_row[::-1] + first_column[::-1]
    elif type == 'kolecko_1_enter_50':
        COORDINATE_X = 15
        COORDINATE_Y = 14
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1
        restricted_area[1, COORDINATE_Y-2] = 0
        # positions as a tape: first row, last column, last row, first column
        first_row = [(0, i) for i in range(1,COORDINATE_Y-1)]
        last_column = [(i, COORDINATE_Y-1) for i in range(1,COORDINATE_X-1)]
        last_row = [(COORDINATE_X-1, i) for i in range(1,COORDINATE_Y-1)]
        first_column = [(i, 0) for i in range(1,COORDINATE_X-1)]
        all_posible_positions = first_row + last_column + last_row[::-1] + first_column[::-1] + [(1, COORDINATE_Y-2)]
    elif type == 'singleline_2_enter_50':
        COORDINATE_X = 52
        COORDINATE_Y = 1
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'singleline_4_enter_50':
        COORDINATE_X = 54
        COORDINATE_Y = 1
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    elif type == 'square':
        COORDINATE_X = 12
        COORDINATE_Y = 12
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1
        # positions as a tape: first row, last column, last row, first column
        first_row = [(0, i) for i in range(1,COORDINATE_Y-1)]
        last_column = [(i, COORDINATE_Y-1) for i in range(1,COORDINATE_X-1)]
        last_row = [(COORDINATE_X-1, i) for i in range(1,COORDINATE_Y-1)]
        first_column = [(i, 0) for i in range(1,COORDINATE_X-1)]
        print(first_row, last_column, last_row, first_column)
        all_posible_positions = first_row + last_column + last_row.reverse() + first_column.reverse()
    elif type == 'kolecko':
        COORDINATE_X = 9
        COORDINATE_Y = 8
        restricted_area = np.zeros((COORDINATE_X, COORDINATE_Y))
        restricted_area[1:COORDINATE_X-1, 1:COORDINATE_Y-1] = 1
        restricted_area[[0, 0, -1, -1], [0, -1, -1, 0]] = 1
        restricted_area[2:COORDINATE_X-2, 2:COORDINATE_Y-2] = 0
        restricted_area[3:COORDINATE_X-3, 3:COORDINATE_Y-3] = 1
        all_posible_positions = [(i, j) for i in range(COORDINATE_X) for j in range(COORDINATE_Y) if restricted_area[i, j] == 0]
    return COORDINATE_X, COORDINATE_Y, restricted_area, all_posible_positions



import matplotlib.pyplot as plt
import matplotlib.patches as patches


class InteractivePlot:
    def __init__(self):
        self.rectangles = []  # List to store the rectangles
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Click to add 1x1 rectangles')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        # Check if the click is inside the axes
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            rect = patches.Rectangle((x, y), 1, 1, edgecolor='r', facecolor='none')
            self.rectangles.append((x, y))
            self.ax.add_patch(rect)  # Add the rectangle to the plot
            self.fig.canvas.draw()
            print(f'Rectangle added at: ({x}, {y})')
            print(f'Current list of rectangles: {self.rectangles}')

    def show(self):
        plt.show()

# Usage
# interactive_plot = InteractivePlot()
# interactive_plot.show()

# The list of points can be accessed via interactive_plot.points
