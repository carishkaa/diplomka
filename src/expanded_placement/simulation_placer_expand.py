import numpy as np
import argparse
import csv
import json
import plotly.express as px
import pandas as pd
import multiprocess
from pymoo.core.problem import StarmapParallelization
import random
import os
from collections import deque

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import PermutationRandomSampling, Sampling
from pymoo.operators.crossover.ox import OrderCrossover, Crossover, ox, random_sequence
from pymoo.operators.mutation.inversion import InversionMutation, Mutation, inversion_mutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

class InterfaceTileSampling(Sampling):
    def __init__(self, n_interfaces, n_tiles, all_machine_positions_idxs, all_interface_locations_idxs):
        """ PermutationRandomSampling for machines and interfaces separately. """
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles
        self.all_machine_positions_idxs = all_machine_positions_idxs
        self.all_interface_locations_idxs = all_interface_locations_idxs
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), -1)
        for i in range(n_samples):
            # Interfaces locations
            X[i, :self.n_interfaces] = np.random.choice(self.all_interface_locations_idxs, self.n_interfaces, replace=False)

            # Dispensers locations (without interfaces) - we can put machine on empty interface-location (TODO discuss if this is correct)
            available_machine_positions_idsx = np.setdiff1d(self.all_machine_positions_idxs, X[i, :self.n_interfaces])
            X[i, self.n_interfaces:] = np.random.choice(available_machine_positions_idsx, self.n_tiles, replace=False)
        return X
    

class SeparateOrderCrossover(Crossover):
    def __init__(self, n_interfaces, n_tiles, shift=False, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        assert n_var == self.n_interfaces + self.n_tiles

        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            parent_a, parent_b = X[:, i, :]

            # Crossover for interfaces
            start, end = random_sequence(self.n_interfaces)
            interfaces_a = ox(parent_a[:self.n_interfaces], parent_b[:self.n_interfaces], seq=(start, end), shift=self.shift)
            interfaces_b = ox(parent_b[:self.n_interfaces], parent_a[:self.n_interfaces], seq=(start, end), shift=self.shift)

            # Crossover for tiles
            start, end = random_sequence(self.n_tiles)
            tiles_a = ox(parent_a[self.n_interfaces:], parent_b[self.n_interfaces:], seq=(start, end), shift=self.shift)
            tiles_b = ox(parent_a[self.n_interfaces:], parent_b[self.n_interfaces:], seq=(start, end), shift=self.shift)

            # Check if there are duplicates
            duplicated_a = set(interfaces_a) & set(tiles_a)
            duplicated_b = set(interfaces_b) & set(tiles_b)
            for d in duplicated_a:
                if len(interfaces_a) > self.n_interfaces:
                    interfaces_a = np.delete(interfaces_a, np.where(interfaces_a == d))
                elif len(tiles_a) > self.n_tiles:
                    tiles_a = np.delete(tiles_a, np.where(tiles_a == d))
                else:
                    raise ValueError("Cannot remove element from parent_a") # TODO in this case add random empty location
            
            for d in duplicated_b:
                if len(interfaces_b) > self.n_interfaces:
                    interfaces_b = np.delete(interfaces_b, np.where(interfaces_b == d))
                elif len(tiles_b) > self.n_tiles:
                    tiles_b = np.delete(tiles_b, np.where(tiles_b == d))
                else:
                    raise ValueError("Cannot remove element from parent_b") # TODO in this case add random empty location
            
            assert len(interfaces_a) >= self.n_interfaces
            assert len(interfaces_b) >= self.n_interfaces
            assert len(tiles_a) >= self.n_tiles
            assert len(tiles_b) >= self.n_tiles
            offspring_a = np.concatenate([interfaces_a[:self.n_interfaces], tiles_a[:self.n_tiles]])
            assert len(offspring_a) == len(parent_a) == len(parent_b)
            offspring_b = np.concatenate([interfaces_b[:self.n_interfaces], tiles_b[:self.n_tiles]])
            assert len(offspring_b) == len(parent_a) == len(parent_b)

            assert len(np.unique(offspring_a)) == len(offspring_a)
            assert len(np.unique(offspring_b)) == len(offspring_b)

            Y[0, i, :] = offspring_a
            Y[1, i, :] = offspring_b

        return Y
    
class MachinesInversionMutation(Mutation):
    def __init__(self, n_interfaces, n_tiles, prob=1.0):
        """Randomly selects a segment of a chromosome and reverse its order. Applied to machines only."""
        super().__init__()
        self.prob = prob
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, y in enumerate(X):
            if np.random.random() < self.prob:
                seq = random_sequence(self.n_tiles)
                Y[i, self.n_interfaces:] = inversion_mutation(y[self.n_interfaces:], seq, inplace=True)
                assert len(np.unique(Y[i])) == len(Y[i])

        return Y

class ExpandedPlacementProblem(ElementwiseProblem):
    def __init__(self, patients, n_tiles, sorted_names, drug_packing, n_interfaces, n_episodes, distances, **kwargs):
        self.n_tiles = n_tiles
        self.sorted_names = sorted_names
        self.n_interfaces = n_interfaces
        self.n_episodes = n_episodes
        self.interface_indices = np.array(range(n_interfaces))

        self.distances = distances

        # reindex required medicines by offset of dispensers - interfaces on the beginning, then dispensers
        self.patients = patients
        reindexed_packing = {}
        for id, drug_list in drug_packing.items():
            reindexed_packing[n_interfaces+int(id)] = drug_list
        self.drug_packing = reindexed_packing
        print('reindexed_packing', reindexed_packing)


        self.reverse_drug_packing = {}
        for drug_name in sorted_names:
            self.reverse_drug_packing[drug_name] = self.compatible_dispenser_list(drug_name)
        print('reverse_drug_packing', self.reverse_drug_packing)

        super().__init__(n_var=n_tiles+n_interfaces, n_obj=1, vtype=int, **kwargs)


    def compatible_dispenser_list(self, drug_name):
        """ Returns a list of dispenser indices that contain the drug_name """
        location_keys = [k for k in self.drug_packing.keys() if drug_name in self.drug_packing[k]]
        assert len(location_keys) >= 1
        return location_keys

    def sample_from_pdf(self, pdf):
        normalized_pdf = [p / sum(pdf) for p in pdf]
        return random.choices(range(len(pdf)), weights=normalized_pdf)[0]

    def _evaluate(self, x, out, *args, **kwargs):
        interface_locations = x[:self.n_interfaces]

        # simulate patients
        total_distance_patients = 0
        for patient in self.patients:
            #print("---- new patient ----")

            for _ in range(self.n_episodes):
                # uniformly random select interface to start
                interface_start_idx = np.random.randint(0, self.n_interfaces)
                start_interface_loc = interface_locations[interface_start_idx]
                prev_loc = start_interface_loc

                drugs_to_dispense = set(list(patient))
                #print(f"drug to dispense: {drugs_to_dispense}")
                distance_for_patient = 0
                while len(drugs_to_dispense) > 0:

                    compatible_dispensers = [] # idxs of chromosome
                    for drug_name in drugs_to_dispense:
                        compatible_dispensers += self.reverse_drug_packing[drug_name] # chromosome: drug = idx, location = element

                    #print(f"compatible dispensers: {compatible_dispensers}")
                    compatible_locations = x[compatible_dispensers]
                    #print(f"site locations {compatible_locations}")

                    distances_to_sites = [1/(self.distances[prev_loc][loc] + 0.1) for loc in compatible_locations]        # to avoid division by zero
                    #print(f"distances to locations {distances_to_sites}")
                    sampled_idx = self.sample_from_pdf(distances_to_sites)  # warning: it is an index to compatible_dispensers array
                    
                    machine_idx = compatible_dispensers[sampled_idx]
                    new_loc = compatible_locations[sampled_idx]
                    # drugs available at sampled location
                    drugs_available_at_location = set(self.drug_packing[machine_idx])
                    #print(f"going to site: {compatible_dispensers[sampled_idx]} containing {drugs_available_at_location}")

                    # multiple drugs might available at the location, we need to pick which to remove from patient
                    remaining_drugs = drugs_to_dispense & drugs_available_at_location       # aka intersection
                    assert len(remaining_drugs) > 0
                    drugs_to_dispense.remove(remaining_drugs.pop())
                    distance_for_patient += self.distances[prev_loc][new_loc]
                    prev_loc = new_loc

                # interface to finish
                distances_to_sites = [1/(self.distances[prev_loc][loc] + 0.1) for loc in interface_locations]
                interface_finish_idx = self.sample_from_pdf(distances_to_sites)
                finish_interface_loc = interface_locations[interface_finish_idx]
                distance_for_patient += self.distances[prev_loc][finish_interface_loc]

                total_distance_patients += distance_for_patient

        out["F"] = total_distance_patients/(self.n_episodes*len(self.patients))
        # TODO constraints (all adjacent)


class ObjValCallback(Callback):
    def __init__(self, n_patients=None, n_episodes=None) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["mean"] = []
        self.data["solution"] = []
        self.n_patients = n_patients
        self.n_episodes = n_episodes

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())
        self.data["mean"].append(algorithm.pop.get("F").mean())

        min_idx = min(range(len(algorithm.pop.get("X"))), key=lambda idx: algorithm.pop.get("F")[idx])
        self.data["solution"].append(algorithm.pop.get("X")[min_idx])


EMPTY_LOCATION = -1
BLOCKED_LOCATION = -2

def format_placement(solution, layout_size, all_available_positions, blocked_locations):
    placement = np.full(layout_size, EMPTY_LOCATION)

    for idx, location in enumerate(solution):
        position = all_available_positions[location]
        placement[position[0], position[1]] = idx

    for blocked_location in blocked_locations:
        placement[blocked_location[0], blocked_location[1]] = BLOCKED_LOCATION

    return placement

def get_neighbors(position, available_machine_positions):
    x, y = position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    neighbors = []
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if [nx, ny] in available_machine_positions:
            neighbors.append((nx, ny))
    return neighbors

def bfs_shortest_paths(start, available_machine_positions):
    start_tuple = tuple(start)
    queue = deque([(start_tuple, 0)])
    distances = {start_tuple: 0}
    
    while queue:
        current, dist = queue.popleft()
        for neighbor in get_neighbors(current, available_machine_positions):
            if neighbor not in distances:  # not visited yet
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    return distances

def compute_shortest_path_matrix(available_machine_positions):
    distance_matrix = np.full((len(available_machine_positions), len(available_machine_positions)), np.inf)
    
    for start_idx, start in enumerate(available_machine_positions):
        shortest_paths = bfs_shortest_paths(start, available_machine_positions)
        for end_idx, end in enumerate(available_machine_positions):
            distance_matrix[start_idx][end_idx] = shortest_paths.get(tuple(end), float('inf'))
    
    return distance_matrix

def compute_placement(args, topology, patient_list, sorted_names, drug_packing):
    n_episodes = args.episodes
    n_interfaces = args.interfaces
    n_popsize = args.pop_size
    n_evals = args.evals

    # topology
    layout_size = (topology["n"], topology["m"])
    blocked_locations = topology["unavailable_locations"]

    interface_locations = topology["interface_locations"]
    assert len(interface_locations) >= n_interfaces, "Not enough available positions for interfaces"
    for x, y in interface_locations:
        assert x < layout_size[0] and y < layout_size[1] and x >= 0 and y >= 0, f"Possible interface location {(x, y)} is out of layout bounds"
        assert (x, y) not in blocked_locations, f"Possible interface location {(x, y)} is blocked"

    all_available_positions = [[x, y] for x in range(layout_size[0]) for y in range(layout_size[1]) if [x, y] not in blocked_locations]
    assert len(all_available_positions) >= n_interfaces + n_tiles, "Not enough available positions for interfaces and dispensers"

    # chromosome representation: [loc_interface1, loc_interface2, ..., loc_interfaceN, loc_dispenser1, loc_dispenser2, ..., loc_dispenserM]
    all_available_position_idxs = list(range(len(all_available_positions)))
    interface_position_idxs = [all_available_positions.index(loc) for loc in interface_locations]

    distances = compute_shortest_path_matrix(all_available_positions)

    pool = multiprocess.Pool(args.processes)
    runner = StarmapParallelization(pool.starmap)
    problem = ExpandedPlacementProblem(patient_list, n_tiles=n_tiles, distances=distances, sorted_names=sorted_names,
                               drug_packing=drug_packing, n_interfaces=n_interfaces, n_episodes=n_episodes,
                               elementwise_runner=runner)

    sampling = InterfaceTileSampling(n_interfaces, n_tiles, all_available_position_idxs, interface_position_idxs)
    termination = get_termination("n_eval", n_evals)

    algorithm = GA(
        pop_size=n_popsize,
        sampling=sampling,  # binary random sampling
        crossover=SeparateOrderCrossover(n_interfaces, n_tiles),  # single point crossover
        mutation=MachinesInversionMutation(n_interfaces, n_tiles),  # bitflip mutation
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   callback=ObjValCallback(n_patients=len(patient_list), n_episodes=n_episodes),
                   # seed=0,
                   verbose=True,
                   elementwise_evaluation=True)
    pool.close()

    # TODO bug!!!: res.X sometimes contains duplicates (interface and machine on the same location)

    # res.X = [loc_idx_interface1, ..., loc_idx_interfaceN, loc_idx_dispenser1, ..., loc_idx_dispenserM]
    # positions = all_available_positions[loc_idx_interface1]
    placement = format_placement(res.X, layout_size, all_available_positions, blocked_locations)
    return placement, res


def save_placement(placement, best_obj, mean_obj, args, checkpoint=None):
    drug_input = json.load(open(args.packing, "r"))
    drug_packing = drug_input["packing"]
    n_tiles = len(drug_packing.keys())

    x = range(len(placement[0]))
    y = range(len(placement))

    medicine_labels = []
    sorted_names_with_interfaces = {EMPTY_LOCATION: "empty", BLOCKED_LOCATION: "blocked"}
    for interface in range(args.interfaces):
        sorted_names_with_interfaces[interface] = "interface"
    for idx, drugs in drug_packing.items():
        sorted_names_with_interfaces[int(idx) + args.interfaces] = ",".join(drugs)
    for yi in y:
        medicine_labels += [[sorted_names_with_interfaces[placement[yi, xi]] for xi in x]]
    placement_colors = np.zeros(shape=(len(y), len(x)))

    def find_index_packed_drug(drug_names):
        for i, d in drug_packing.items():
            if set(d) == set(drug_names):
                return i
        if drug_names[0] == "interface":
            return -1
        if drug_names[0] == "blocked":
            return BLOCKED_LOCATION
        return None

    for i in x:
        for j in y:
            drugs_name = medicine_labels[j][i]
            placement_colors[j, i] = find_index_packed_drug(drugs_name.split(","))

    if args.figs:
        max_color_ratio = np.nanmax(placement_colors.flatten())
        min_color_ratio = np.nanmin(placement_colors.flatten())
        def normalize_color(color_ratio):
            return (color_ratio - min_color_ratio) / (max_color_ratio - min_color_ratio)
        interface_color = normalize_color(-1)
        start_medicine_color = normalize_color(0)
        one_part = (1 - start_medicine_color) / (len(px.colors.sequential.Jet)-1)
        jet_colors = [[start_medicine_color + one_part * i, px.colors.sequential.Jet[i]] for i in range(len(px.colors.sequential.Jet))]
        custom_colors2 = [[0, "black"], [interface_color, "white"]] + jet_colors + [[1, px.colors.sequential.Jet[-1]]]
        
        fig = px.imshow(placement_colors, title=f"expected number of steps: {min(best_obj)}",
                        color_continuous_scale=custom_colors2)
        fig.update_traces(customdata=medicine_labels, hovertemplate='%{customdata}')
        if checkpoint is not None:
            fig.write_html(os.path.join(args.output, f"topology_{args.topology}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_placer_simulation_checkpoints_{checkpoint}.html"))
        else:
            fig.write_html(os.path.join(args.output, f"topology_{args.topology}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_placer_simulation.html"))

        fig_convergence = px.line(pd.DataFrame({"best": best_obj, "mean": mean_obj}), labels=dict(x="generation [-]", y="expected step count [-]"))
        if checkpoint is not None:
            fig_convergence.write_html(os.path.join(args.output, f"topology_{args.topology}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_convergence_plot_checkpoints_{checkpoint}.html"))
        else:
            fig_convergence.write_html(os.path.join(args.output, f"topology_{args.topology}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_convergence_plot.html"))

    topology_export = {
        "n_tiles": n_tiles,
        "n_interfaces": args.interfaces,
        "n_process": args.processes,
        "n_episodes": args.episodes,
        "n_evals": args.evals,
        "n_popsize": args.pop_size,
        "obj_progress": {"mean": mean_obj, "best": best_obj},
        "obj": min(best_obj),
        "placement": medicine_labels,
        "packer_result": drug_packing
    }

    filename = f"topology_TODO_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_ndispensers_{drug_input['n_dispensers']}_nevals_{args.evals}.json"
    if len(args.output) > 0:
        filename = os.path.join(args.output, filename)

    if checkpoint is not None:
        filename = filename[0:-5] + f"_checkpoint_{checkpoint}.json"

    json.dump(topology_export,
              open(filename, "w+"),
              indent=4
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimizes dispenser placement.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-p", "--packing", help="packer json file", type=str)
    parser.add_argument("-t", "--topology", help="json file with topology info: n, m, blocked locations, interface locations", type=str) 
    parser.add_argument("-i", "--interfaces", help="number of interface locations", type=int, default=2)
    parser.add_argument("-o", "--output", help="output directory for placement", type=str, default="")
    parser.add_argument("--evals", help="maximum number of evaluations", type=int, default=30000)
    parser.add_argument("--pop-size", help="size of population", type=int, default=100)
    parser.add_argument("--episodes", help="number of simulation episodes per patient", type=int, default=5)
    parser.add_argument("--processes", help="number of processes for optimization", type=int, default=10)
    parser.add_argument("--save-figs", help="save option for placement figure",  dest='figs', default=False)
    parser.add_argument("--save-checkpoints", help="save intermediate checkpoints along the solution process", dest='checkpoints', default=False)

    args = parser.parse_args()
    topology = json.load(open(args.topology, "r"))

    drug_input = json.load(open(args.packing, "r"))
    sorted_names = drug_input["sorted_names"]
    drug_packing = drug_input["packing"]

    n_medicines = len(sorted_names)
    n_tiles = len(drug_packing.keys())

    # load real patients
    patient_list = []
    with open("generated_capsules_with_dosages.csv", "r") as generated_capsules:
        reader = csv.reader(generated_capsules, delimiter=";")
        next(reader)
        for line in reader:
            requested_drugs = line[1][2:-2].split("', '")
            patient_list += [set(requested_drugs)]

    placement, res = compute_placement(args, topology, patient_list, sorted_names, drug_packing)

    if args.checkpoints:
        solutions = res.algorithm.callback.data["solution"]
        best_obj = res.algorithm.callback.data["best"]
        mean_obj = res.algorithm.callback.data["mean"]

        first_solution = format_placement(solutions[0], args.topology, n_tiles, args.interfaces)
        save_placement(first_solution, [best_obj[0]], [mean_obj[0]], args, checkpoint=0)
        prev_obj = best_obj[0]

        for sol_id in range(1, len(best_obj)):
            if best_obj[sol_id] < prev_obj:
                placement = format_placement(solutions[sol_id], args.topology, n_tiles, args.interfaces)
                save_placement(placement, best_obj[0:(sol_id+1)], mean_obj[0:(sol_id+1)], args, checkpoint=sol_id)
                prev_obj = best_obj[sol_id]
    else:
        save_placement(placement, res.algorithm.callback.data["best"], res.algorithm.callback.data["mean"], args)
