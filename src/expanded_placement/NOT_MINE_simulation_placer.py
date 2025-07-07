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

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.display.output import Output, Column


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


class PlacementProblem(ElementwiseProblem):
    def __init__(self, patients, n_tiles, topology, sorted_names, drug_packing, n_interfaces, n_episodes, **kwargs):
        self.n_tiles = n_tiles
        self.topology = topology
        self.n_medicines = len(sorted_names)
        self.sorted_names = sorted_names
        self.n_interfaces = n_interfaces
        self.n_episodes = n_episodes
        self.interface_indices = np.array(range(n_interfaces))

        assert self.topology in ['line', 'doubleline', 'ring', 'square']
        if self.topology in ['doubleline', 'ring']:  # double line and ring topology should contain an even number of tiles (2(a+b) dimensions)
            assert (self.n_tiles + self.n_interfaces) % 2 == 0
        if self.topology == 'square':  # square topology needs to have n^2 tiles
            assert np.abs(
                int(np.sqrt(self.n_tiles + self.n_interfaces)) - np.sqrt(self.n_tiles + self.n_interfaces)) < 0.01

        # reindex required medicines by offset of dispensers
        self.patients = patients
        reindexed_packing = {}
        for id, drug_list in drug_packing.items():
            reindexed_packing[int(id)+n_interfaces] = drug_list
        self.drug_packing = reindexed_packing
        print('reindexed_packing', reindexed_packing)


        self.reverse_drug_packing = {}
        for drug_name in sorted_names:
            self.reverse_drug_packing[drug_name] = self.compatible_dispenser_list(drug_name)


        print('reverse_drug_packing', self.reverse_drug_packing)

        super().__init__(n_var=n_tiles+n_interfaces, n_obj=1, vtype=int, **kwargs)


    def compatible_dispenser_list(self, drug_name):
        location_keys = [k for k in self.drug_packing.keys() if drug_name in self.drug_packing[k]]
        assert len(location_keys) >= 1
        return location_keys

    def find_locations(self, location_list, placement):
        return [np.argwhere(placement == dispenser_id)[0] for dispenser_id in location_list]

    def distance_between(self, from_idx, to_idx):
        if self.topology in ['doubleline', 'line', 'square']:
            return np.linalg.norm(from_idx-to_idx, 1)
        if self.topology == 'ring':
            return min([np.linalg.norm(from_idx-to_idx, 1),
                        1+np.linalg.norm([0]-from_idx, 1) + np.linalg.norm(to_idx-[self.n_tiles+self.n_interfaces-1], 1),
                        1+np.linalg.norm([0]-to_idx, 1) + np.linalg.norm(from_idx-[self.n_tiles+self.n_interfaces-1], 1)
                        ])

    def sample_from_pdf(self, pdf):
        # Normalize the PDF to ensure it sums to 1
        total = sum(pdf)
        normalized_pdf = [p / total for p in pdf]

        # Sample from the PDF using weights
        return random.choices(range(len(pdf)), weights=normalized_pdf)[0]

    def _evaluate(self, x, out, *args, **kwargs):
        # reconstruct representation
        if self.topology == 'doubleline':
            placement = np.reshape(x, (2, (self.n_tiles+self.n_interfaces) // 2))
        elif self.topology == 'square':
            sq_size = int(np.sqrt(self.n_tiles+self.n_interfaces))
            placement = np.reshape(x, (sq_size, sq_size))
        elif self.topology == 'line' or self.topology == 'ring':
            placement = x
        else:
            print('not implemented yet')
            exit(0)

        interface_locations = np.argwhere(placement < self.n_interfaces)

        # simulate patients
        total_distance_patients = 0
        for patient in self.patients:
            #print("---- new patient ----")

            for i in range(self.n_episodes):
                # uniformly random select interface to start
                interface_start_idx = np.random.randint(0, self.n_interfaces)
                start_interface_loc = interface_locations[interface_start_idx]
                last_loc = start_interface_loc

                drugs_to_dispense = set(list(patient))
                #print(f"drug to dispense: {drugs_to_dispense}")
                distance_for_patient = 0
                while len(drugs_to_dispense) > 0:

                    compatible_dispensers = []
                    for drug_name in drugs_to_dispense:
                        compatible_dispensers += self.reverse_drug_packing[drug_name]

                    #print(f"compatible dispensers: {compatible_dispensers}")
                    compatible_locations = self.find_locations(compatible_dispensers, placement)
                    #print(f"site locations {compatible_locations}")

                    distances_to_sites = [1/(self.distance_between(last_loc, loc) + 0.1) for loc in compatible_locations]        # to avoid division by zero
                    #print(f"distances to locations {distances_to_sites}")
                    sampled_idx = self.sample_from_pdf(distances_to_sites)  # warning: it is an index to compatible_dispensers array
                    location_idx = compatible_dispensers[sampled_idx]
                    new_loc = compatible_locations[sampled_idx]
                    # drugs available at sampled location
                    drugs_available_at_location = set(self.drug_packing[location_idx])
                    #print(f"going to site: {compatible_dispensers[sampled_idx]} containing {drugs_available_at_location}")

                    # multiple drugs might available at the location, we need to pick which to remove from patient
                    remaining_drugs = drugs_to_dispense & drugs_available_at_location       # aka intersection
                    assert len(remaining_drugs) > 0
                    drugs_to_dispense.remove(remaining_drugs.pop())
                    distance_for_patient += self.distance_between(last_loc, new_loc)
                    last_loc = new_loc

                # interface to finish
                compatible_locations = self.find_locations(list(range(self.n_interfaces)), placement)
                distances_to_sites = [1/(self.distance_between(last_loc, loc) + 0.1) for loc in compatible_locations]
                # sample finish interface wrt. distance from last location
                interface_finish_idx = self.sample_from_pdf(distances_to_sites)
                finish_interface_loc = interface_locations[interface_finish_idx]
                # add it to the traveled distance for the patient simulated
                distance_for_patient += self.distance_between(last_loc, finish_interface_loc)

                total_distance_patients += distance_for_patient

        out["F"] = total_distance_patients/(self.n_episodes*len(self.patients))


def format_placement(solution, topology, n_tiles, n_interfaces):
    if topology == 'doubleline':
        placement = np.reshape(solution, (2, (n_tiles + n_interfaces) // 2))
    elif topology in ['line', 'ring']:
        placement = solution
    elif topology == 'square':
        square_size = int(np.sqrt(n_tiles + n_interfaces))
        placement = np.reshape(solution, (square_size, square_size))
    return placement


def compute_placement(args, patient_list, sorted_names, drug_packing):

    n_episodes = args.episodes
    n_interfaces = args.interfaces
    topology = args.topology
    n_popsize = args.pop_size
    n_evals = args.evals

    pool = multiprocess.Pool(args.processes)
    runner = StarmapParallelization(pool.starmap)
    problem = PlacementProblem(patient_list, n_tiles=n_tiles, topology=topology, sorted_names=sorted_names,
                               drug_packing=drug_packing, n_interfaces=n_interfaces, n_episodes=n_episodes,
                               elementwise_runner=runner)

    termination = get_termination("n_eval", n_evals)

    algorithm = GA(
        pop_size=n_popsize,
        sampling=PermutationRandomSampling(),  # binary random sampling
        crossover=OrderCrossover(),  # single point crossover
        mutation=InversionMutation(),  # bitflip mutation
        eliminate_duplicates=True
    )

    # class MyOutput(Output):
    #     def __init__(self):
    #         super().__init__()
    #         self.x_mean = Column("x_mean", width=13)
    #         self.x_std = Column("x_std", width=13)
    #         self.columns += [self.x_mean, self.x_std]

    #     def update(self, algorithm):
    #         super().update(algorithm)
    #         self.x_mean.set(np.mean(algorithm.pop.get("X")))
    #         self.x_std.set(np.std(algorithm.pop.get("X")))

    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   callback=ObjValCallback(n_patients=len(patient_list), n_episodes=n_episodes),
                   # seed=0,
                   verbose=True,
                #    output=MyOutput(),
                   elementwise_evaluation=True)
    pool.close()

    placement = format_placement(res.X, topology, n_tiles, n_interfaces)

    #if topology == 'doubleline':
    #    placement = np.reshape(res.X, (2, (n_tiles + n_interfaces) // 2))
    #elif topology in ['line', 'ring']:
    #    placement = res.X
    #elif topology == 'square':
    #    square_size = int(np.sqrt(n_tiles + n_interfaces))
    #    placement = np.reshape(res.X, (square_size, square_size))

    return placement, res

def save_placement(placement, best_obj, mean_obj, args, checkpoint=None):
    drug_input = json.load(open(args.packing, "r"))
    drug_packing = drug_input["packing"]
    n_tiles = len(drug_packing.keys())

    if args.topology in ['line', 'ring']:
        placement_line = np.zeros(shape=(1, len(placement)))
        placement_line[0][:] = placement
        placement = placement_line

    x = range(len(placement[0]))
    y = range(len(placement))

    medicine_labels = []
    sorted_names_with_interfaces = {}
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
        return None

    for i in x:
        for j in y:
            drugs_name = medicine_labels[j][i]
            placement_colors[j, i] = find_index_packed_drug(drugs_name.split(","))

    if args.figs:
        fig = px.imshow(placement_colors, title=f"expected number of steps: {min(best_obj)}",
                        color_continuous_scale='jet')
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
        "topology": args.topology,
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

    filename = f"topology_{args.topology}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_ndispensers_{drug_input['n_dispensers']}_nevals_{args.evals}.json"
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
    parser.add_argument("-t", "--topology", help="type of topology (line/doubleline/ring/square)", default="line")
    parser.add_argument("-i", "--interfaces", help="number of interface locations", type=int, default=2)
    parser.add_argument("-o", "--output", help="output directory for placement", type=str, default="")
    parser.add_argument("--evals", help="maximum number of evaluations", type=int, default=30000)
    parser.add_argument("--pop-size", help="size of population", type=int, default=100)
    parser.add_argument("--episodes", help="number of simulation episodes per patient", type=int, default=5)
    parser.add_argument("--processes", help="number of processes for optimization", type=int, default=10)
    parser.add_argument("--save-figs", help="save option for placement figure",  dest='figs', default=False)
    parser.add_argument("--save-checkpoints", help="save intermediate checkpoints along the solution process", dest='checkpoints', default=False)

    args = parser.parse_args()

    assert args.topology in ["line", "doubleline", "ring", "square"]

    drug_input = json.load(open(args.packing, "r"))
    sorted_names = drug_input["sorted_names"]
    drug_packing = drug_input["packing"]

    n_medicines = len(sorted_names)
    n_tiles = len(drug_packing.keys())

    # load real patients
    patient_list = []
    with open("src/expanded_placement/generated_capsules_with_dosages_200.csv", "r") as generated_capsules:
        reader = csv.reader(generated_capsules, delimiter=";")
        next(reader)
        for line in reader:
            requested_drugs = line[1][2:-2].split("', '")
            patient_list += [set(requested_drugs)]

    placement, res = compute_placement(args, patient_list, sorted_names, drug_packing)

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

# n_tiles ... number of tiles 
# n_interfaces ... number of interfaces 
# total = n_tiles + n_interfaces