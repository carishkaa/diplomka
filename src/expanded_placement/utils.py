# Storing here some functions that i don't need anymore in case i need them later
from plotly.subplots import make_subplots
def plot_init_pop(format_placement, get_medicine_labels, get_placement_colors, get_color_scale, args, layout: Layout, drug_packing, res):
    layout_size = (layout["n"], layout["m"])
    blocked_locations = layout["unavailable_locations"]
    all_available_positions = [(x, y) for x in range(layout_size[0]) for y in range(layout_size[1]) if [x, y] not in blocked_locations]
    if args.figs:
        initial_population = res.algorithm.callback.data["initial_population"]
        fitnesses = res.algorithm.callback.data["initial_population_fitnesses"]
        rows = 10
        cols = int(len(initial_population)/rows)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"id {i+1}; {np.round(fitnesses[i], 1)}" for i in range(len(initial_population))])
        for idx, individual in enumerate(initial_population):
            initial_placement = format_placement(individual, layout, all_available_positions)
            medicine_labels = get_medicine_labels(initial_placement, args, drug_packing, range(layout_size[1]), range(layout_size[0]))
            placement_colors = get_placement_colors(drug_packing, range(layout_size[1]), range(layout_size[0]), medicine_labels)
            row = idx // cols + 1
            col = idx % cols + 1
            fig.add_trace(px.imshow(placement_colors, color_continuous_scale=get_color_scale(placement_colors)).data[0], row=row, col=col)

        fig.update_layout(coloraxis=dict(colorscale=get_color_scale(placement_colors)), height=1700, width=1500)
        fig.write_html(os.path.join(args.output, "initial_population.html"))



# MOVING REPAIR

# array of dicts that contain 'left', 'right', 'up', 'down' keys with valuues = arrays of all the avail positions on in this line on this direction (e.g. for (0,1) it will be {'left': [], 'right': [(1,1), (2,1), ...], 'up': [(0,2), (0,3)], 'down': [(0,0)]}
    # self.direction_positions = []
    # for idx, pos in enumerate(all_available_positions):
    #     x, y = pos
    #     down = [self.reverse_coord_to_idx[(x, y_)] for y_ in range(y) if (x, y_) in all_available_positions][::-1]
    #     up = [self.reverse_coord_to_idx[(x, y_)] for y_ in range(y+1, layout["m"]) if (x, y_) in all_available_positions] # right
    #     left = [self.reverse_coord_to_idx[(x_, y)] for x_ in range(x) if (x_, y) in all_available_positions][::-1]
    #     right = [self.reverse_coord_to_idx[(x_, y)] for x_ in range(x+1, layout["n"]) if (x_, y) in all_available_positions]
    #     self.direction_positions.append({'left': left, 'right': right, 'up': up, 'down': down})

# 2. Repair holes - identify where there is more machines+interfaces - up/down/left/right and move the opposite side to the hole
            # while True: 
            #     current_hole = holes.pop(0) if len(holes) > 0 else None
            #     if current_hole is None:
            #         break
                
            #     assert current_hole not in individual
            #     hole = current_hole
            #     x, y = self.all_available_positions[hole]
            #     # find where is more machines+interfaces
            #     directions = ['left', 'right', 'up', 'down']
            #     opposite_directions = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
            #     directions_count = {}
            #     directions_count_interfaces = {}
            #     for direction in directions:
            #         all_pos = self.direction_positions[hole][direction]
            #         directions_count[direction] = len(set(all_pos) & set(individual))
            #         directions_count_interfaces[direction] = len(set(all_pos) & set(individual[:self.n_interfaces]))

            #     directions_without_interfaces = {k: v for k, v in directions_count.items() if v > 0 and directions_count_interfaces[k] == 0}
            #     # directions_count_without_0 = {k: v for k, v in directions_count.items() if v > 0}
            #     if len(directions_without_interfaces) == 0:
            #         # TODO if directions_without_interfaces empty -> pick random machine that is adjacent to empty location and move it to the hole
            #         # pick random machine that is adjacent to empty location and move it to the hole
            #         # while True:
            #         #     random_machine = np.random.choice(individual[self.n_interfaces:])
            #         #     x_, y_ = self.all_available_positions[random_machine]
            #         #     neigh = get_neighbors((x_, y_), self.all_available_positions)
            #         print("Repair: No directions without interfaces")
            #         continue

            #     min_direction = min(directions_without_interfaces, key=directions_without_interfaces.get)
            #     min_direction_positions = self.direction_positions[hole][min_direction]
            #     min_direction_positions_used = individual[np.isin(individual, min_direction_positions)]
            #     # min_direction_positions_used and not in individual[:self.n_interfaces]
            #     min_direction_positions_used_and_not_interface = np.setdiff1d(min_direction_positions_used, individual[:self.n_interfaces]) 

            #     # move all of them one step towards the hole
            #     min_direction_positions_coords = [self.all_available_positions[pos] for pos in min_direction_positions_used_and_not_interface]

            #     # 0,6 up: 0,7 ; 0,8 => y-1
            #     new_positions = []

            #     def find_one_step_closer_position(x__, y__, direction):
            #         while True:
            #             if direction == 'left':
            #                 x__ -= 1
            #             elif direction == 'right':
            #                 x__ += 1
            #             elif direction == 'up':
            #                 y__ += 1
            #             elif direction == 'down':
            #                 y__ -= 1
            #             assert x__ >= 0 and x__ < layout["n"] and y__ >= 0 and y__ < layout["m"], "Out of bounds"
            #             if (x__, y__) in self.all_available_positions:
            #                 break
            #         return self.reverse_coord_to_idx[(x__, y__)]

            #     for pos in min_direction_positions_coords:
            #         x_, y_ = pos
            #         new_positions.append(find_one_step_closer_position(x_, y_, opposite_directions[min_direction]))

            #     # idxs of individual where individual == min_direction_positions
            #     individual[np.where(np.isin(individual, min_direction_positions_used_and_not_interface))] = new_positions
            #     assert len(np.unique(individual)) == len(individual), "Duplicates in individual" # 

            #     # check if any hole is in changed area (holes) FIXME: this is not working
            #     copy_holes = holes.copy()
            #     for hole in copy_holes:
            #         if hole in min_direction_positions: # todo not in min direction, but in opposite
            #             hole_coord = self.all_available_positions[hole]
            #             new_hole = find_one_step_closer_position(hole_coord[0], hole_coord[1], opposite_directions[min_direction])
            #             assert new_hole not in individual, "New hole is already filled"
            #             holes.remove(hole)
            #             holes.append(new_hole)
