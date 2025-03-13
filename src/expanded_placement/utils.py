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
