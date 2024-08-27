#define the problem
import itertools
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.core.callback import Callback

class Floorplanning(ElementwiseProblem):
    def __init__(self, permutation, machines, capsules, distances, layout, frequences):
        self.permutation = permutation
        self.capsules = capsules
        # matrix of co-occurences
        self.freq = frequences
        self.distances = distances
        self.layout = layout
        self.machines = machines
        super().__init__(n_var=len(permutation), n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = evaluate_between_machines(self, x, self.machines, self.distances, self.layout, self.capsules)

def evaluate_between_machines(self, x, machines, distances, layout, capsules):
    sum_of_all_capsules_distances = 0
    if layout == 'square' or layout == 'rectangle':
        for recept in capsules:
            min_dist, _ = calc_capsule_distance_oneway_traffic(x, recept, distances)
            sum_of_all_capsules_distances += min_dist
    elif layout == 'square_1_enter_1_exit':
        for recept in capsules:
            min_dist, enter_exit_idxs = calc_capsule_distance_oneway_traffic(x, recept, distances)
            sum_of_all_capsules_distances += min_dist
            # Add distance to entrance and exit
            enter_idx, exit_idx = enter_exit_idxs["begin"], enter_exit_idxs["end"]
            sum_of_all_capsules_distances += min(
                self.distances_to_entrance[enter_idx] + self.distances_to_exit[exit_idx], 
                self.distances_to_entrance[exit_idx] + self.distances_to_exit[enter_idx]
            )
    elif layout == 'doubeline_1_enter_50':
        for capsule in capsules:
            min_dist, capsule_end_indxs = calc_capsule_distance_oneway_traffic_with_dupes(x, machines, capsule, distances, self.freq)
            sum_of_all_capsules_distances += min_dist

            # print(min_dist)
            enter_machine_idx, exit_machine_idx = capsule_end_indxs["begin"], capsule_end_indxs["end"]

            # self.distances_to_exit contains distance of several exits for each machine. For each combinations of enter/exit, we need to find the minimum and add it to the sum
            sum_of_all_capsules_distances += min(
                self.distances_to_exit[:, enter_machine_idx].min(), 
                self.distances_to_exit[:, exit_machine_idx].min()
            )
    elif layout == 'doubeline_2_enter_50' or layout == 'doubeline_4_enter_50' or layout == 'kolecko_2_enter_50' or layout == 'kolecko_4_enter_50':
        # print('eval')
        # distances_to_exit[exit_idx, machine_idx]
        chromosome = machines[x.astype(int)]
        outlet_idxs = np.where(chromosome == -10)[0]
        distances_to_exit = np.zeros((outlet_idxs.__len__(), machines.__len__()))
        for i, outlet_idx in enumerate(outlet_idxs):
            for j, machine in enumerate(machines):
                distances_to_exit[i, j] = distances[outlet_idx, machine]
        # print(distances_to_exit)

        for capsule in capsules:
            # min_dist = calc_best_placement_with_dupes(x, machines, capsule, distances, self.freq)
            # sum_of_all_capsules_distances += min_dist
            min_dist, capsule_end_indxs = calc_capsule_distance_oneway_traffic_with_dupes(x, machines, capsule, distances, self.freq)
            sum_of_all_capsules_distances += min_dist
            enter_machine_idx, exit_machine_idx = capsule_end_indxs["begin"], capsule_end_indxs["end"]

            # self.distances_to_exit contains distance of several exits for each machine. For each combinations of enter/exit, we need to find the minimum and add it to the sum
            sum_of_all_capsules_distances += min(
                distances_to_exit[:, enter_machine_idx].min(), 
                distances_to_exit[:, exit_machine_idx].min()
            )

    else:
        for recept in capsules:
            min_dist = calc_capsule_distance_overall_combinations(x, recept, distances)
            sum_of_all_capsules_distances += min_dist
    return sum_of_all_capsules_distances

class RecordBsfCosts(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["bsf_costs"] = []
        self.data["n_gen"] = []

    def notify(self, algorithm):
        self.data["n_gen"].append(algorithm.n_gen)
        self.data["bsf_costs"].append(algorithm.pop.get("F"))
        if algorithm.n_gen % 100 == 0:
            print(f"Generation: {algorithm.n_gen} | Best cost: {algorithm.pop.get('F').min()}")

def create_manhattan_distances_matrix(positions: list):
    num_of_positions = len(positions)
    distances = np.zeros((num_of_positions, num_of_positions))
    for i in range(num_of_positions):
        for j in range(num_of_positions):
            distances[i, j] = np.abs(positions[i][0] - positions[j][0]) + np.abs(positions[i][1] - positions[j][1])
    return distances

def calc_capsule_distance_overall_combinations(curr_x: np.ndarray, recept: np.ndarray, distances: np.ndarray):
    cur_position_indexes = np.where(np.isin(curr_x, recept[recept != 0]))[0]
    # Min distance of perm
    all_perms_of_recept = list(itertools.permutations(cur_position_indexes))
    distances_of_perms = np.zeros(len(all_perms_of_recept))
    for ind, perm in enumerate(all_perms_of_recept):
        dist = sum([distances[perm[i], perm[i+1]] for i in range(len(perm) - 1)])
        distances_of_perms[ind] = dist
    min_dist = np.min(distances_of_perms)
    return min_dist

def calc_capsule_distance_oneway_traffic(curr_x: np.ndarray, recept: np.ndarray, distances: np.ndarray):
    """
    Calculate the minimum distance of the current permutation of the capsules to the recept
    @return `cur_position_indexes`: the sorted indexes of the capsules in the current permutation
    """
    cur_position_indexes = np.where(np.isin(curr_x, recept[recept != 0]))[0]
    cur_position_indexes = np.sort(cur_position_indexes)
    drugs_num = len(cur_position_indexes)
    all_dist = [distances[cur_position_indexes[i], cur_position_indexes[(i+1)%drugs_num]] for i in range(drugs_num)]
    assert all_dist.__len__() == drugs_num

    idx_of_largest = np.argmax(all_dist)
    dists_without_largest = np.delete(all_dist, idx_of_largest)
    # dists_without_largest = np.sort(all_dist)[:-1] if all_dist.__len__() > 1 else all_dist
    min_sum_dist = np.sum(dists_without_largest)

    begin, end = cur_position_indexes[idx_of_largest], cur_position_indexes[(idx_of_largest+1)%drugs_num]

    return min_sum_dist, dict(begin=begin, end=end)

def calc_capsule_distance_oneway_traffic_with_dupes(curr_x: np.ndarray, machines: np.ndarray, recept: np.ndarray, distances: np.ndarray, freq):
    """
    Calculate the minimum distance of the current permutation of the capsules to the recept
    @return `cur_position_indexes`: the sorted indexes of the capsules in the current permutation
    """
    # on which position which drug is placed
    machines_x = machines[curr_x.astype(int)]
    # print('recept', recept)
    # print('machines_x', machines_x)

    # [i][j], i is drug idx
    position_idxs_drug = [np.where(np.isin(machines_x, i))[0] for i in recept[recept != -1]]
    capsule_idxs = np.where(recept != -1)[0]
    # print('position_idxs_drug', position_idxs_drug)

    positions_combinations = list(itertools.product(*position_idxs_drug))
    # print(positions_combinations)
    min_sum_dist = np.inf
    result_idx_of_largest = -1
    result_cur_position_indexes = []
    for cur_position_indexes in positions_combinations:
        # print('==', cur_position_indexes)
        cur_position_indexes = np.sort(cur_position_indexes)
        # print('cur_position_indexes', cur_position_indexes)
        drugs_num = len(cur_position_indexes)
        all_dist = [distances[cur_position_indexes[i], cur_position_indexes[(i+1)%drugs_num]] for i in range(drugs_num)]
        all_freq = [freq[capsule_idxs[i], capsule_idxs[(i+1)%drugs_num]] for i in range(drugs_num)]
        assert all_dist.__len__() == drugs_num
        assert all_freq.__len__() == drugs_num

        idx_of_largest = np.argmax(all_dist)
        dists_without_largest = np.delete(all_dist, idx_of_largest)
        freqs_without_largest = np.delete(all_freq, idx_of_largest)
        # dists_without_largest = np.sort(all_dist)[:-1] if all_dist.__len__() > 1 else all_dist

        # sum of dist * freq
        # potential_min_sum_dist = np.sum(dists_without_largest)
        potential_min_sum_dist = np.sum(np.multiply(dists_without_largest, freqs_without_largest))

        if potential_min_sum_dist < min_sum_dist:
            min_sum_dist = potential_min_sum_dist
            result_idx_of_largest = idx_of_largest
            result_cur_position_indexes = cur_position_indexes

    # print(result_cur_position_indexes, result_idx_of_largest)
    begin, end = result_cur_position_indexes[result_idx_of_largest], result_cur_position_indexes[(result_idx_of_largest+1)%drugs_num]
    # todo also consider combinations WITH exit/entrance 
    return min_sum_dist, dict(begin=begin, end=end)


def calc_best_placement_with_dupes(curr_x: np.ndarray, machines: np.ndarray, recept: np.ndarray, distances: np.ndarray, freq):
    """
    ALL possible combinations (really slow...)
    """
    capsule_idxs = np.where(recept != -1)[0]
    # print('capsule' , capsule_idxs.__len__())
    # if capsule_idxs.__len__() > 5: # skip if more than 6 drugs in capsule for now (too slow)
    #     return 0

    # on which position which drug is placed
    machines_x = machines[curr_x.astype(int)] # todo consifer that -10 is outlet.

    # [i][j], i is drug idx
    position_idxs_drug = [np.where(np.isin(machines_x, i))[0] for i in recept[recept != -1]] # all the machines positions of this drug
    # add to position_idxs_drug all the exits

    positions_combinations = list(itertools.product(*position_idxs_drug)) # all possible combinations of how to fill this capsule

    # all possible permutations of positions_combinations
    positions_combinations_permutations = [list(itertools.permutations(i)) for i in positions_combinations]
    positions_combinations_permutations = list(itertools.chain(*positions_combinations_permutations))
    # print('positions_combinations_permutations', positions_combinations_permutations)

    # Create all possible pairs of inlets and outlets
    position_idxs_inoutlets = np.where(machines_x == -10)[0]
    inlets_outlets_combinations = list(itertools.product(position_idxs_inoutlets, repeat=2))

    # For positions_combinations_permutations, add inlets_outlets_combinations to each of them
    all_combinations = []
    for pos_comb in positions_combinations_permutations:
        for in_out in inlets_outlets_combinations:
            all_combinations.append([in_out[0]] + list(pos_comb) + [in_out[1]])

    min_sum_dist = np.inf
    for cur_position_indexes in all_combinations:
        # print('cur_position_indexes', cur_position_indexes)
        drugs_in_capsule_num = len(capsule_idxs)
        all_dist = [distances[cur_position_indexes[i], cur_position_indexes[i+1]] for i in range(len(cur_position_indexes)-1)]
        all_freq = [1] + [freq[capsule_idxs[i], capsule_idxs[i+1]] for i in range(drugs_in_capsule_num-1)] + [1]
        assert all_dist.__len__() == len(cur_position_indexes)-1 
        assert all_freq.__len__() == drugs_in_capsule_num + 1
        assert all_dist.__len__() == all_freq.__len__()

        # sum of dist * freq
        potential_min_sum_dist = np.sum(np.multiply(all_dist, all_freq))

        if potential_min_sum_dist < min_sum_dist:
            min_sum_dist = potential_min_sum_dist

    return min_sum_dist
