from itertools import islice
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
from datetime import datetime
import time
from typing import TypedDict
import time
import networkx as nx

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import Sampling
from pymoo.operators.crossover.ox import Crossover, ox, random_sequence
from pymoo.operators.mutation.inversion import Mutation, inversion_mutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column


class InterfaceTileSampling(Sampling):
    def __init__(self, n_interfaces, n_tiles, all_machine_positions_idxs, all_interface_locations_idxs, all_available_positions: list[tuple[int, int]], percent_of_random_perm: float):
        """ PermutationRandomSampling for machines and interfaces separately. """
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles
        self.all_machine_positions_idxs = all_machine_positions_idxs
        self.all_interface_locations_idxs = all_interface_locations_idxs
        self.all_available_positions = all_available_positions

        assert percent_of_random_perm >= 0 and percent_of_random_perm <= 1, "percent_of_random_perm should be in range [0, 1]"
        self.percent_of_random_perm = percent_of_random_perm
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), -1)
        num_random_samples = int(n_samples * self.percent_of_random_perm)

        # Random permutations
        for i in range(num_random_samples):
            # Interfaces locations
            X[i, :self.n_interfaces] = np.random.choice(self.all_interface_locations_idxs, self.n_interfaces, replace=False)

            # Dispensers locations (without interfaces) - we can put machine on empty interface-location (TODO discuss if this is correct)
            available_machine_positions_idsx = np.setdiff1d(self.all_machine_positions_idxs, X[i, :self.n_interfaces])
            X[i, self.n_interfaces:] = np.random.choice(available_machine_positions_idsx, self.n_tiles, replace=False)
            assert len(np.unique(X[i])) == len(X[i])

        # Connected
        for i in range(num_random_samples, n_samples):
            start_loc = np.random.choice(self.all_interface_locations_idxs, 1)[0]
            connected_area = [start_loc]
            interfaces_num = 1
            interface_locations = [start_loc]
            machines_num = 0
            machine_locations = []

            # Warning: this can be infinite loop if it will not find enough interfaces connected to current solution area
            while interfaces_num < self.n_interfaces or machines_num < self.n_tiles:
                connect_to = np.random.choice(connected_area, 1)[0]
                neighbors_coord = get_neighbors(self.all_available_positions[connect_to], self.all_available_positions)
                neighbors_loc = [self.all_available_positions.index(n) for n in neighbors_coord]
                neighbors_empty = np.setdiff1d(neighbors_loc, connected_area)

                if len(neighbors_empty) == 0:
                    continue
                random_neighbor = np.random.choice(neighbors_empty, 1)[0]
                if random_neighbor in self.all_interface_locations_idxs and interfaces_num < self.n_interfaces:
                    connected_area.append(random_neighbor)
                    interface_locations.append(int(random_neighbor))
                    interfaces_num += 1
                elif machines_num < self.n_tiles:
                    connected_area.append(random_neighbor)
                    machine_locations.append(int(random_neighbor))
                    machines_num += 1
            X[i, :self.n_interfaces] = interface_locations
            X[i, self.n_interfaces:] = np.random.permutation(machine_locations)
            assert len(np.unique(X[i])) == len(X[i])
        return X

class SeparateOrderCrossover(Crossover):
    def __init__(self, n_interfaces, n_tiles, all_machine_positions_idxs, all_interface_locations_idxs, shift=False, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles
        self.all_machine_positions_idxs = all_machine_positions_idxs
        self.all_interface_locations_idxs = all_interface_locations_idxs

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        assert n_var == self.n_interfaces + self.n_tiles

        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            parent_a, parent_b = X[:, i, :]

            # Crossover for interfaces TODO maybe for interfaces we don't need order crossover, better try smth else?
            start, end = random_sequence(self.n_interfaces)
            interfaces_a = ox(parent_a[:self.n_interfaces], parent_b[:self.n_interfaces], seq=(start, end), shift=self.shift)
            interfaces_b = ox(parent_b[:self.n_interfaces], parent_a[:self.n_interfaces], seq=(start, end), shift=self.shift)

            # Crossover for tiles
            start, end = random_sequence(self.n_tiles)
            tiles_a = ox(parent_a[self.n_interfaces:], parent_b[self.n_interfaces:], seq=(start, end), shift=self.shift)
            tiles_b = ox(parent_a[self.n_interfaces:], parent_b[self.n_interfaces:], seq=(start, end), shift=self.shift)

            # Check if there are duplicates
            def repair_duplicates(interfaces, tiles):
                duplicated = set(interfaces) & set(tiles)
                for d in duplicated:
                    if len(interfaces) > self.n_interfaces:
                        interfaces = np.delete(interfaces, np.where(interfaces == d))
                    elif len(tiles) > self.n_tiles:
                        tiles = np.delete(tiles, np.where(tiles == d))
                    else:
                        random_empty_location = np.random.choice(np.setdiff1d(self.all_machine_positions_idxs, np.concatenate([interfaces, tiles])), 1)
                        if random_empty_location in self.all_interface_locations_idxs:
                            index = np.where(interfaces == d)
                            interfaces[index] = random_empty_location
                        else:
                            index = np.where(tiles == d)
                            tiles[index] = random_empty_location
                return interfaces, tiles

            interfaces_a, tiles_a = repair_duplicates(interfaces_a, tiles_a)
            interfaces_b, tiles_b = repair_duplicates(interfaces_b, tiles_b)

            assert len(interfaces_a) >= self.n_interfaces
            assert len(interfaces_b) >= self.n_interfaces
            assert len(tiles_a) >= self.n_tiles
            assert len(tiles_b) >= self.n_tiles

            np.random.shuffle(interfaces_a)
            np.random.shuffle(interfaces_b)

            offspring_a = np.concatenate([interfaces_a[:self.n_interfaces], tiles_a[:self.n_tiles]])
            assert len(offspring_a) == len(parent_a) == len(parent_b)
            assert len(np.unique(offspring_a)) == len(offspring_a), "Duplicates in offspring A"

            offspring_b = np.concatenate([interfaces_b[:self.n_interfaces], tiles_b[:self.n_tiles]])
            assert len(offspring_b) == len(parent_a) == len(parent_b)
            assert len(np.unique(offspring_b)) == len(offspring_b), "Duplicates in offspring B"

            Y[0, i, :] = offspring_a
            Y[1, i, :] = offspring_b
        return Y

class MachinesMutation(Mutation):
    def __init__(self, n_interfaces, n_tiles, all_available_positions: list[tuple[int, int]], prob=1.0, **kwargs):
        """
        Applies to machines only (second part of chromosome). Inverse sequence, swap and move operators.
        """
        super().__init__()
        self.prob = prob
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles
        self.all_available_positions = all_available_positions

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        current_gen = kwargs.get('algorithm').n_gen

        # TODO check what operators do we need and with what probability
        for i, y in enumerate(X):
            # Inverse sequence
            if np.random.random() < 0.7:
                seq = random_sequence(self.n_tiles)
                Y[i, self.n_interfaces:] = inversion_mutation(y[self.n_interfaces:], seq, inplace=True)
                assert len(np.unique(Y[i])) == len(Y[i])
            
            swap_proba = 0.3 if current_gen < 100 else 0.9
            # Swap - exchange two machines
            if np.random.random() < swap_proba: # this is good in the end
                idx = np.random.choice(range(self.n_tiles))
                new_loc = np.random.choice(np.setdiff1d(range(self.n_tiles), [idx]))
                Y[i, self.n_interfaces + idx], Y[i, self.n_interfaces + new_loc] = Y[i, self.n_interfaces + new_loc], Y[i, self.n_interfaces + idx]
                assert len(np.unique(Y[i])) == len(Y[i])

            # Move one machine to empty location adjacent to any other machine (on X or Y axis)
            # move_proba = 0.5 if current_gen < 50 else 0.1
            if np.random.random() < 0: # this is good in the beginning to try different locations of area
                idx = np.random.choice(range(self.n_tiles))
                position = self.all_available_positions[Y[i, self.n_interfaces + idx]]

                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                neighbors_coord = get_neighbors(position, self.all_available_positions, directions)
                neighbors = [self.all_available_positions.index(n) for n in neighbors_coord if n in self.all_available_positions]
                empty_neighbors = np.setdiff1d(neighbors, Y[i])

                if len(empty_neighbors) > 0:
                    # new_idx = np.random.choice(empty_neighbors)
                    # prefer moving to the location adjacent to smth else
                    neigh_adjacencies_count = np.zeros(len(empty_neighbors)) # empty_neighbors + current location
                    for idx, empty_idx in enumerate(empty_neighbors):
                        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            x, y = self.all_available_positions[empty_idx]
                            nx, ny = x + direction[0], y + direction[1]
                            if [nx, ny] in self.all_available_positions and self.all_available_positions.index([nx, ny]) in Y[i]:
                                neigh_adjacencies_count[idx] += 1
                    cur_adjacencies_count = 0
                    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        x, y = position
                        nx, ny = x + direction[0], y + direction[1]
                        if [nx, ny] in self.all_available_positions and self.all_available_positions.index([nx, ny]) in Y[i]:
                            cur_adjacencies_count += 1
                    if cur_adjacencies_count == 0 or np.max(neigh_adjacencies_count) > cur_adjacencies_count:
                        Y[i, self.n_interfaces + idx] = empty_neighbors[np.argmax(neigh_adjacencies_count)]
                    else:
                        Y[i, self.n_interfaces + idx] = Y[i, self.n_interfaces + idx]
                assert len(np.unique(Y[i])) == len(Y[i])
        return Y
    
class Layout(TypedDict):
    n: int
    m: int
    unavailable_locations: list[tuple[int, int]]
    interface_locations: list[tuple[int, int]]

class RepairHoles(Repair):
    # TODO maybe repair not only holes, but use it as a mutation and try it for all empty locations - move most distant machine to this empty locations 
    def __init__(self, layout: Layout, n_interfaces, n_tiles, all_available_positions_idxs, all_interface_locations_idxs, all_available_positions: list[tuple[int, int]], blocked_positions, distances):
        self.n_interfaces = n_interfaces
        self.n_tiles = n_tiles
        self.all_avail_positions_idxs = all_available_positions_idxs
        self.all_available_positions = all_available_positions
        self.blocked_positions = blocked_positions
        self.distances = distances

        self.reverse_coord_to_idx = {pos: idx for idx, pos in enumerate(all_available_positions)}
        super().__init__()

    def is_hole(self, empty_loc, individual):
        x, y = self.all_available_positions[empty_loc]
        present_neighbors = get_neighbors((x, y), [self.all_available_positions[i] for i in individual])
        blocked_neighbors = get_neighbors((x, y), self.blocked_positions)
        is_boundary = x == 0 or x == self.all_available_positions[-1][0] or y == 0 or y == self.all_available_positions[-1][1]
        is_corner = is_boundary and ((x == 0 and y == 0) or (x == 0 and y == self.all_available_positions[-1][1]) or (x == self.all_available_positions[-1][0] and y == 0) or (x == self.all_available_positions[-1][0] and y == self.all_available_positions[-1][1]))
        if len(present_neighbors) == 4 \
                or (len(present_neighbors) == 3 and is_boundary) \
                or (len(present_neighbors) == 3 and len(blocked_neighbors) == 1) \
                or (len(present_neighbors) == 2 and is_corner)\
                or (len(present_neighbors) == 2 and is_boundary and len(blocked_neighbors) == 1)\
                or (len(present_neighbors) == 2 and len(blocked_neighbors) == 2):
            return True
        return False
    
    def repair_holes(self, old_individual, holes):
        individual = old_individual.copy()
        if len(holes) > 0:
            distances_interface2tile = np.zeros((self.n_interfaces, self.n_tiles))
            for i, interface in enumerate(individual[:self.n_interfaces]):
                for j, machine in enumerate(individual[self.n_interfaces:]):
                    distances_interface2tile[i, j] = self.distances[interface][machine]
            min_dist_interface2tile = np.min(distances_interface2tile, axis=0)

            while True:
                current_hole = holes.pop(0) if len(holes) > 0 else None
                if current_hole is None:
                    break

                # Find machine that is most distant from interfaces TODO consider checking distance from current hole 
                max_dist_idx = np.argmax(min_dist_interface2tile)
                individual[self.n_interfaces + max_dist_idx] = current_hole
                assert len(np.unique(individual)) == len(individual), "Duplicates in individual"

                # Recalculate distances_interface2tile and min_dist_interface2tile
                for i, interface in enumerate(individual[:self.n_interfaces]):
                    distances_interface2tile[i, max_dist_idx] = self.distances[interface][individual[self.n_interfaces + max_dist_idx]]
                min_dist_interface2tile = np.min(distances_interface2tile, axis=0)
        return individual

    def _do(self, problem, X: np.ndarray, **kwargs):
        # if kwargs.get('algorithm').n_gen < 100:
        #     return X
       
        for _, individual in enumerate(X):
            empty_locations = np.setdiff1d(self.all_avail_positions_idxs, individual)
            holes = []
            for empty_loc in empty_locations:
                assert empty_loc not in individual # TODO remove, used for debugging
                if self.is_hole(empty_loc, individual):
                    holes.append(empty_loc)
            X[_, :] = self.repair_holes(individual, holes)
            assert len(np.unique(X[_, :])) == len(X[_, :]), "Duplicates in individual after holes repair" # TODO remove?, used for debugging


            empty_locations = np.setdiff1d(self.all_avail_positions_idxs, individual)
            lonely_dispensers = []
            for dispenser_id in range(self.n_interfaces, self.n_interfaces + self.n_tiles):
                x, y = self.all_available_positions[individual[dispenser_id]]
                present_neighbors = get_neighbors((x, y), [self.all_available_positions[i] for i in individual])
                if len(present_neighbors) == 0:
                    lonely_dispensers.append(dispenser_id)

            empty_adjacent = [n for n in empty_locations if len(get_neighbors(self.all_available_positions[n], [self.all_available_positions[i] for i in individual])) > 0]
            dist_interface2emptyAdj = np.zeros((self.n_interfaces, len(empty_adjacent)))
            for i, interface in enumerate(individual[:self.n_interfaces]):
                for j, empty in enumerate(empty_adjacent):
                    dist_interface2emptyAdj[i, j] = self.distances[interface][empty]
            min_dist_interface2emptyAdj = np.min(dist_interface2emptyAdj, axis=0)
            while len(lonely_dispensers) > 0:
                dispenser_id = lonely_dispensers.pop(0)
                individual[dispenser_id] = empty_adjacent[np.argmin(min_dist_interface2emptyAdj)]
                assert len(np.unique(individual)) == len(individual), "Duplicates in individual after loners repair"

                # remove from empty_adjacent, recalculate distances
                empty_adjacent = np.setdiff1d(empty_adjacent, [individual[dispenser_id]])
                dist_interface2emptyAdj = np.delete(dist_interface2emptyAdj, np.argmin(min_dist_interface2emptyAdj), axis=1)
                min_dist_interface2emptyAdj = np.min(dist_interface2emptyAdj, axis=0)

            X[_, :] = individual
            assert len(np.unique(X[_, :])) == len(X[_, :]), "Duplicates in individual after loners repair"

        return X


class ExpandedPlacementProblem(ElementwiseProblem):
    def __init__(self, patients: list[set], n_tiles, sorted_names, drug_packing, n_interfaces, n_episodes, distances: np.ndarray, drug_dosing, all_available_position_idxs, all_available_positions, **kwargs):
        self.n_tiles = n_tiles
        self.sorted_names = sorted_names
        self.n_interfaces = n_interfaces
        self.n_episodes = n_episodes
        self.interface_indices = np.array(range(n_interfaces))
        self.drug_dosing = drug_dosing

        self.distances = distances

        # reindex required medicines by offset of dispensers - interfaces on the beginning, then dispensers
        self.patients = patients
        reindexed_packing = {}
        for id, drug_list in drug_packing.items():
            reindexed_packing[n_interfaces+int(id)] = set(drug_list)
        self.drug_packing = reindexed_packing
        # print('reindexed_packing', reindexed_packing)

        self.reverse_drug_packing = {}
        for drug_name in sorted_names:
            self.reverse_drug_packing[drug_name] = self.compatible_dispenser_list(drug_name)
        # print('reverse_drug_packing', self.reverse_drug_packing)

        self.all_available_positions = all_available_positions

        super().__init__(n_var=n_tiles+n_interfaces, n_obj=1, vtype=int, **kwargs)

    def compatible_dispenser_list(self, drug_name):
        """ Returns a list of dispenser indices that contain the drug_name """
        location_keys = [k for k in self.drug_packing.keys() if drug_name in self.drug_packing[k]]
        assert len(location_keys) >= 1
        return location_keys

    def sample_from_pdf(self, pdf):
        return random.choices(range(len(pdf)), weights=pdf)[0]

    def _evaluate(self, x, out, *args, **kwargs):
        # starttime = time.perf_counter() # TODO remove - tracks time of evaluation
        x_loc_to_idx_map = {val: idx for idx, val in enumerate(x)}

        # processing_count = np.zeros(self.n_tiles+self.n_interfaces, dtype=int)
        processing_time = np.zeros(self.n_tiles+self.n_interfaces, dtype=int)

        interface_locations = x[:self.n_interfaces]

        # simulate patients
        total_distance_patients = 0

        # to track all pairs of dispensers/interfaces that were visited by patients
        simulation_pairs_count = np.zeros((len(x), len(x)), dtype=int)

        interface_start_idxs = np.random.choice(range(self.n_interfaces), self.n_episodes * len(self.patients)) # pregenerating

        for idx_p, patient in enumerate(self.patients):
            for e in range(self.n_episodes):
                # Randomly select interface to start
                interface_start_idx = interface_start_idxs[idx_p * self.n_episodes + e]
                prev_loc = interface_locations[interface_start_idx]
                # processing_count[interface_start_idx] += 1
                processing_time[interface_start_idx] += 1

                drugs_to_dispense = set(patient)
                distance_for_patient = 0
                while len(drugs_to_dispense) > 0:

                    compatible_dispensers = [] # idxs of chromosome
                    for drug_name in drugs_to_dispense:
                        compatible_dispensers += self.reverse_drug_packing[drug_name] # chromosome: drug = idx, location = element

                    compatible_locations = x[compatible_dispensers]
                    distances_to_sites = 1/(self.distances[prev_loc][compatible_locations] + 0.1)        # to avoid division by zero

                    sampled_idx = self.sample_from_pdf(distances_to_sites)  # warning: it is an index to compatible_dispensers array
                    sampled_dispenser = compatible_dispensers[sampled_idx]
                    cur_loc = x[sampled_dispenser]

                    simulation_pairs_count[x_loc_to_idx_map[prev_loc]][x_loc_to_idx_map[cur_loc]] += 1

                    drugs_available_at_location: set = self.drug_packing[sampled_dispenser]

                    # multiple drugs might available at the location, we need to pick which to remove from patient
                    remaining_drugs = drugs_to_dispense & drugs_available_at_location       # aka intersection
                    assert len(remaining_drugs) > 0
                    current_drug = remaining_drugs.pop()
                    # processing_count[sampled_dispenser] += 1
                    processing_time[sampled_dispenser] += self.drug_dosing[current_drug]
                    drugs_to_dispense.remove(current_drug)
                    distance_for_patient += self.distances[prev_loc][cur_loc]
                    prev_loc = cur_loc

                # interface to finish
                distances_to_sites = 1/(self.distances[prev_loc][interface_locations] + 0.1)
                interface_finish_idx = self.sample_from_pdf(distances_to_sites)
                finish_interface_loc = interface_locations[interface_finish_idx]
                distance_for_patient += self.distances[prev_loc][finish_interface_loc]
                # processing_count[interface_finish_idx] += 1
                processing_time[interface_finish_idx] += 1
                simulation_pairs_count[x_loc_to_idx_map[prev_loc]][x_loc_to_idx_map[finish_interface_loc]] += 1

                total_distance_patients += distance_for_patient

        # Make simulation_pairs a triangle matrix by summing elements at (i, j) and (j, i)
        for i in range(simulation_pairs_count.shape[0]):
            for j in range(i + 1, simulation_pairs_count.shape[1]):
                simulation_pairs_count[i, j] += simulation_pairs_count[j, i]
                simulation_pairs_count[j, i] = 0

        # endtime = time.perf_counter()
        # print_debug(f"Time taken in ms: {(endtime - starttime) * 1000:.2f} ms")
        out["F"] = total_distance_patients/(self.n_episodes*len(self.patients))
        # return

        out["F_expected_steps"] = total_distance_patients/(self.n_episodes*len(self.patients))
        # out["processing_count"] = processing_count
        out["processing_time"] = processing_time / self.n_episodes # avg processing time per episode

        # starttime = time.perf_counter()

        # Dispensers with processing time > bound_processing_time are considered overprocessed
        bound_processing_time = np.quantile(out["processing_time"], 0.6)

        # starttime_allpairs = time.perf_counter()
        # Filter simulation_pairs to have only pairs that contains at least one interface OR at least one overprocessed dispenser
        all_pairs = dict()
        for i in range(len(simulation_pairs_count)):
            for j in range(i + 1, len(simulation_pairs_count)):
                if simulation_pairs_count[i][j] > 0:
                    if i < self.n_interfaces or j < self.n_interfaces or out["processing_time"][i] > bound_processing_time or out["processing_time"][j] > bound_processing_time:
                        pair = tuple(sorted((x[i], x[j])))
                        if pair not in all_pairs:
                            all_pairs[pair] = 0
                        all_pairs[pair] += simulation_pairs_count[i][j]
        simulation_pairs_count = None
        # endtime_allpairs = time.perf_counter()
        # print_debug(f"Time taken in ms (allpairs): {(endtime_allpairs - starttime_allpairs) * 1000:.2f} ms")

        # starttime_creategraph = time.perf_counter()
        # Keep only valid locations (not empty, not overprocessed, not interface)
        dispensers_loc_coords = [self.all_available_positions[loc] for loc in x[self.n_interfaces:]]
        overprocessing_locations = x[np.where(out["processing_time"] > bound_processing_time)[0]]
        overprocessing_loc_coords = [self.all_available_positions[loc] for loc in overprocessing_locations]
        valid_loc_coords = [loc for loc in self.all_available_positions if loc in dispensers_loc_coords and loc not in overprocessing_loc_coords]
        valid_loc_coords = set(valid_loc_coords)
        G = create_grid_graph(valid_loc_coords)
        # endtime_creategraph = time.perf_counter()
        # print_debug(f"Time taken in ms (creategraph): {(endtime_creategraph - starttime_creategraph) * 1000:.2f} ms")

        # starttime_checkpaths = time.perf_counter()
        non_interrupted_pairs = 0 # we want to maximize this
        for pair, pair_count in all_pairs.items():
            A, B = pair
            A_coord = self.all_available_positions[A]
            B_coord = self.all_available_positions[B]

            is_A_valid = A_coord in valid_loc_coords
            is_B_valid = B_coord in valid_loc_coords

            if not is_A_valid:
                G.add_node(A_coord)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (A_coord[0] + dx, A_coord[1] + dy)
                    if neighbor in valid_loc_coords or neighbor == B_coord:
                        G.add_edge(A_coord, neighbor)

            if not is_B_valid:
                G.add_node(B_coord)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (B_coord[0] + dx, B_coord[1] + dy)
                    if neighbor in valid_loc_coords or neighbor == A_coord:
                        G.add_edge(B_coord, neighbor)

            # Is there a path that does not go through empty/overprocessed locations and interfaces?
            len_without_overprocessed = shortest_path_len(G, A_coord, B_coord) # 20 ms
            if len_without_overprocessed != -1:
                non_interrupted_pairs += pair_count
            if not is_A_valid:
                G.remove_node(A_coord)
            if not is_B_valid:
                G.remove_node(B_coord)
        # endtime_checkpaths = time.perf_counter()
        # print_debug(f"Time taken in ms (checkpaths): {(endtime_checkpaths - starttime_checkpaths) * 1000:.2f} ms")

        total_pair_count = sum(all_pairs.values())
        interrupted_pairs = total_pair_count - non_interrupted_pairs # we want to minimize this
        out["F_interruptions"] = interrupted_pairs * 0.0007

        # endtime = time.perf_counter()
        # print_debug(f"Time taken in ms: {(endtime - starttime) * 1000:.2f} ms")

        out["F"] += interrupted_pairs * 0.01

def shortest_path_len(G, source, target):
    """ -1 if no path exists, otherwise returns length of the path """
    try:
        p = nx.bidirectional_shortest_path(G, source, target)
        return len(p) - 1
    except nx.NetworkXNoPath:
        return -1

print_debugy = True
def print_debug(*args, **kwargs):
    if print_debugy:
        print(*args, **kwargs)

class ObjValCallback(Callback):
    def __init__(self, n_patients=None, n_episodes=None) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["mean"] = []
        self.data["processing_count"] = []
        self.data["processing_time"] = []
        self.data["solution"] = []

        self.n_patients = n_patients
        self.n_episodes = n_episodes
        # self.data["initial_population"] = None
        # self.data["initial_population_fitnesses"] = None

    def notify(self, algorithm):
        # if self.data["initial_population"] is None:
        #     self.data["initial_population"] = algorithm.pop.get("X")
        #     self.data["initial_population_fitnesses"] = algorithm.pop.get("F")

        self.data["best"].append(algorithm.pop.get("F").min())
        self.data["mean"].append(algorithm.pop.get("F").mean())

        best_idx = np.argmin(algorithm.pop.get("F"))
        min_idx = min(range(len(algorithm.pop.get("X"))), key=lambda idx: algorithm.pop.get("F")[idx]) # TODO remove
        assert best_idx == min_idx
        self.data["processing_count"].append(algorithm.pop.get("processing_count")[best_idx])
        self.data["processing_time"].append(algorithm.pop.get("processing_time")[best_idx])
        self.data["solution"].append(algorithm.pop.get("X")[best_idx])

class MyOutput(Output):
    def __init__(self):
        super().__init__()
        self.f_avg = Column("f_avg", width=13)
        self.f_min = RoundedColumn("f_min", width=13, ndigits_round=5)
        self.f_interruption = RoundedColumn("f_interruptions", width=15, ndigits_round=5)
        self.F_expected_steps = RoundedColumn("f_expected_steps", width=15, ndigits_round=5)
        self.columns += [self.f_avg, self.f_min, self.f_interruption, self.F_expected_steps]

    def update(self, algorithm):
        super().update(algorithm)
        self.f_avg.set(round(algorithm.pop.get("F").mean(), 4))
        argmin_idx = np.argmin(algorithm.pop.get("F"))
        self.f_min.set(algorithm.pop.get("F")[argmin_idx][0])
        self.f_interruption.set(round(algorithm.pop.get("F_interruptions")[argmin_idx], 3))
        self.F_expected_steps.set(algorithm.pop.get("F_expected_steps")[argmin_idx])

class RoundedColumn(Column):
    def __init__(self, name, width=13, func=None, truncate=True, ndigits_round=4) -> None:
        super().__init__(name, width=width, func=func, truncate=truncate)
        self.ndigits_round = ndigits_round
    def text(self):
        value = self.value
        if value is None:
            value = "-"
        value = round(value, self.ndigits_round) if isinstance(value, float) else value
        text = str(value).rjust(self.width)
        return text


def get_neighbors(position, available_machine_positions: list[tuple[int, int]], directions=[(-1, 0), (1, 0), (0, -1), (0, 1)]) -> list[tuple[int, int]]:
    """ Returns list of not unavailable neighbors of given position """
    x, y = position
    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (nx, ny) in available_machine_positions:
            neighbors.append((nx, ny))
    return neighbors

def bfs_shortest_paths(start: tuple, available_machine_positions: list[tuple[int, int]]):
    queue = deque([(start, 0)])
    distances = {start: 0}
    
    while queue:
        current, dist = queue.popleft()
        for neighbor in get_neighbors(current, available_machine_positions):
            if neighbor not in distances:  # not visited yet
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    return distances

# TODO remove in favour of calculation through nx
def compute_shortest_path_matrix(available_machine_positions: list[tuple[int, int]]):
    distance_matrix = np.full((len(available_machine_positions), len(available_machine_positions)), np.inf)
    
    for start_idx, start in enumerate(available_machine_positions):
        shortest_paths = bfs_shortest_paths(start, available_machine_positions)
        for end_idx, end in enumerate(available_machine_positions):
            distance_matrix[start_idx][end_idx] = shortest_paths.get(end, float('inf'))
    return distance_matrix

BLOCKED_LOCATION = -2
EMPTY_LOCATION = -1

def format_placement(solution, layout: Layout, all_available_positions):
    """ Returns nxm matrix with placement of interfaces and dispensers """
    placement = np.full((layout["n"], layout["m"]), EMPTY_LOCATION)
    for idx, location in enumerate(solution):
        position = all_available_positions[location]
        placement[position[0], position[1]] = idx

    for blocked_location in layout["unavailable_locations"]:
        placement[blocked_location[0], blocked_location[1]] = BLOCKED_LOCATION
    return placement

def assert_layout(layout: Layout, n_interfaces: int, n_tiles: int):
    assert len(layout["interface_locations"]) >= n_interfaces, "Not enough available positions for interfaces. Check layout file."
    for x, y in layout["interface_locations"]:
        assert x < layout["n"] and y < layout["m"] and x >= 0 and y >= 0, f"Possible interface location {(x, y)} is out of layout bounds. Check layout file."
        assert (x, y) not in layout["unavailable_locations"], f"Possible interface location {(x, y)} is blocked. Check layout file."
    for x, y in layout["unavailable_locations"]:
        assert x < layout["n"] and y < layout["m"] and x >= 0 and y >= 0, f"Blocked location {(x, y)} is out of layout. Check layout file."
    assert layout["n"] * layout["m"] - len(layout["unavailable_locations"]) >= n_interfaces + n_tiles, "Not enough available positions for interfaces and dispensers. Check layout file."
    duplicate_interfaces = [loc for loc in layout["interface_locations"] if layout["interface_locations"].count(loc) > 1]
    assert len(duplicate_interfaces) == 0, f"Duplicate interface locations: {duplicate_interfaces}. Check layout file."

def create_grid_graph(valid_nodes: list[tuple[int, int]]):
    G = nx.Graph()
    for node in valid_nodes:
        G.add_node(node)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for node in valid_nodes:
        for di, dj in directions:
            neighbor = (node[0] + di, node[1] + dj)
            if neighbor in valid_nodes:
                G.add_edge(node, neighbor)
    return G

def compute_placement(args, layout: Layout, patient_list, sorted_names, drug_packing, drug_dosing):
    n_episodes = args.episodes
    n_interfaces = args.interfaces
    n_popsize = args.pop_size
    n_evals = args.evals

    # Layout
    assert_layout(layout, n_interfaces, n_tiles)
    all_available_positions = [(x, y) for x in range(layout["n"]) for y in range(layout["m"]) if (x, y) not in layout["unavailable_locations"]]
    empty_location_counts = layout["n"] * layout["m"] - len(layout["unavailable_locations"]) - n_interfaces - n_tiles

    # Chromosome representation: [loc_interface1, ..., loc_interfaceN, loc_dispenser1, ..., loc_dispenserM]
    all_available_position_idxs = list(range(len(all_available_positions)))
    interface_position_idxs = [all_available_positions.index(loc) for loc in layout["interface_locations"]]

    # Warning: we don't know here which cells will be empty, so this is not very accurate
    distances = compute_shortest_path_matrix(all_available_positions)

    pool = multiprocess.Pool(args.processes)
    runner = StarmapParallelization(pool.starmap)
    problem = ExpandedPlacementProblem(patient_list, n_tiles=n_tiles, distances=distances, sorted_names=sorted_names,
                               drug_packing=drug_packing, n_interfaces=n_interfaces, n_episodes=n_episodes,
                               elementwise_runner=runner, drug_dosing=drug_dosing, all_available_position_idxs=all_available_position_idxs, all_available_positions=all_available_positions)

    sampling = InterfaceTileSampling(n_interfaces, n_tiles, all_available_position_idxs, interface_position_idxs, all_available_positions, 
                                     percent_of_random_perm=1 if empty_location_counts > 0 else 0.5)
    repair_holes = RepairHoles(layout, n_interfaces, n_tiles, all_available_position_idxs, interface_position_idxs, all_available_positions, layout["unavailable_locations"], distances)
    termination = get_termination("n_eval", n_evals)

    algorithm = GA(
        pop_size=n_popsize,
        sampling=sampling,
        repair=None if empty_location_counts == 0 else repair_holes,
        crossover=SeparateOrderCrossover(n_interfaces, n_tiles, all_available_position_idxs, interface_position_idxs),
        mutation=MachinesMutation(n_interfaces, n_tiles, all_available_positions),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   callback=ObjValCallback(n_patients=len(patient_list), n_episodes=n_episodes),
                   # seed=0,
                   verbose=True,
                   output=MyOutput(),
                   elementwise_evaluation=True)
    pool.close()

    placement = format_placement(res.X, layout, all_available_positions)
    return placement, res

def get_medicine_labels(placement, args, drug_packing, x, y):
    medicine_labels = []
    sorted_names_with_interfaces = {EMPTY_LOCATION: "empty", BLOCKED_LOCATION: "blocked"}
    for interface in range(args.interfaces):
        sorted_names_with_interfaces[interface] = "interface"
    for idx, drugs in drug_packing.items():
        sorted_names_with_interfaces[int(idx) + args.interfaces] = ",".join(drugs)
    for yi in y:
        medicine_labels += [[sorted_names_with_interfaces[placement[yi, xi]] for xi in x]]
    return medicine_labels

def get_placement_colors(drug_packing, x, y, medicine_labels):
    placement_colors = np.zeros(shape=(len(y), len(x)))

    def find_index_packed_drug(drug_names):
        for i, d in drug_packing.items():
            if set(d) == set(drug_names):
                return i
        if drug_names[0] == "empty":
            return EMPTY_LOCATION
        if drug_names[0] == "blocked":
            return BLOCKED_LOCATION
        return None

    for i in x:
        for j in y:
            drugs_name = medicine_labels[j][i]
            placement_colors[j, i] = find_index_packed_drug(drugs_name.split(","))
    return placement_colors

def get_color_scale(placement_colors):
    max_color_ratio = np.nanmax(placement_colors.flatten())
    min_color_ratio = np.nanmin(placement_colors.flatten())
    def normalize_color(location_idx):
        return (location_idx - min_color_ratio) / (max_color_ratio - min_color_ratio)
    blocked_color = normalize_color(BLOCKED_LOCATION)
    empty_color = normalize_color(EMPTY_LOCATION)
    start_medicine_color = normalize_color(0)
    color_seq = px.colors.sequential.Jet
    one_part = (1 - start_medicine_color) / (len(color_seq)-1)
    medicines_colors = [[start_medicine_color + one_part * i, color_seq[i]] for i in range(len(color_seq))]
    return [[blocked_color, "black"], [empty_color, "white"]] + medicines_colors + [[1, color_seq[-1]]]

def save_placement(placement, best_obj, mean_obj, args, checkpoint=None):
    drug_input = json.load(open(args.packing, "r"))
    drug_packing = drug_input["packing"]
    n_tiles = len(drug_packing.keys())

    x = range(len(placement[0]))
    y = range(len(placement))

    medicine_labels = get_medicine_labels(placement, args, drug_packing, x, y)
    placement_colors = get_placement_colors(drug_packing, x, y, medicine_labels)

    layout_string_for_filename = f"layout_{x[-1]+1}x{y[-1]+1}" # TODO maybe smth better to capture blocked and interface locations 
    if args.figs:        
        unique_id_by_date = datetime.now().strftime("%y%m%d%H%M")
        fig = px.imshow(placement_colors, title=f"expected number of steps: {min(best_obj)}",
                        color_continuous_scale=get_color_scale(placement_colors))
        fig.update_layout(
            xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, mirror=True, showline=True, linecolor='lightgray'),
            yaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, mirror=True, showline=True, linecolor='lightgray')
        )
        fig.update_traces(customdata=medicine_labels, hovertemplate='%{customdata}', xgap=0.1, ygap=0.1)
        if checkpoint is not None:
            fig.write_html(os.path.join(args.output, f"{layout_string_for_filename}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_placer_simulation_checkpoints_{checkpoint}_{unique_id_by_date}.html"))
        else:
            fig.write_html(os.path.join(args.output, f"{layout_string_for_filename}_{y[-1]+1}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_placer_simulation_{unique_id_by_date}.html"))

        fig_convergence = px.line(pd.DataFrame({"best": best_obj, "mean": mean_obj}), labels=dict(x="generation [-]", y="expected step count [-]"))
        if checkpoint is not None:
            fig_convergence.write_html(os.path.join(args.output, f"{layout_string_for_filename}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_convergence_plot_checkpoints_{checkpoint}_{unique_id_by_date}.html"))
        else:
            fig_convergence.write_html(os.path.join(args.output, f"{layout_string_for_filename}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_nevals_{args.evals}_convergence_plot_{unique_id_by_date}.html"))

    layout_export = {
        "n_tiles": n_tiles,
        "n_interfaces": args.interfaces,
        "n_process": args.processes,
        "n_episodes": args.episodes,
        "n_evals": args.evals,
        "n_popsize": args.pop_size,
        "obj_progress": {"mean": mean_obj, "best": best_obj},
        "obj": min(best_obj),
        "placement": medicine_labels,
        "packer_result": drug_packing,
        "layout": layout
    }

    filename = f"{layout_string_for_filename}_ntiles_{n_tiles}_ninterfaces_{args.interfaces}_ndispensers_{drug_input['n_dispensers']}_nevals_{args.evals}.json"
    if len(args.output) > 0:
        filename = os.path.join(args.output, filename)

    if checkpoint is not None:
        filename = filename[0:-5] + f"_checkpoint_{checkpoint}.json"

    json.dump(layout_export,
              open(filename, "w+"),
              indent=4
    )

def save_processing_times_plot(placement, processing_count, processing_times, args):
    # store a picture with heatmap of processing counts for each machine and interface
    xs = range(len(placement[0]))
    ys = range(len(placement))

    # processing_counts_matrix = np.zeros((len(placement), len(placement[0])))
    processing_times_matrix = np.zeros((len(placement), len(placement[0])))
    medicine_labels = get_medicine_labels(placement, args, drug_packing, xs, ys)
    for i in ys:
        for j in xs:
            if placement[i, j] == EMPTY_LOCATION:
                # processing_counts_matrix[i, j] = np.nan
                processing_times_matrix[i, j] = np.nan
            elif placement[i, j] == BLOCKED_LOCATION:
                # processing_counts_matrix[i, j] = np.nan
                processing_times_matrix[i, j] = np.nan
            else:
                # processing_counts_matrix[i, j] = processing_count[placement[i, j]]
                processing_times_matrix[i, j] = processing_times[placement[i, j]]

    processing_times_text = processing_times_matrix.copy().astype(str)
    for i in range(len(medicine_labels)):
        for j in range(len(medicine_labels[0])):
            if medicine_labels[i][j] == "interface":
                processing_times_text[i, j] = f"I: {int(processing_times_matrix[i, j])}"
            if medicine_labels[i][j] == "empty":
                processing_times_text[i, j] = "empty"
            if medicine_labels[i][j] == "blocked":
                processing_times_text[i, j] = "blocked"
    

    # fig = px.imshow(processing_counts_matrix, title="Processing count for each machine and interface in the solution",
    #                 color_continuous_scale="Viridis", text_auto=True)
    # fig.update_layout(
    #     xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, mirror=True, showline=True, linecolor='lightgray'),
    #     yaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, mirror=True, showline=True, linecolor='lightgray')
    # )
    # fig.update_traces(customdata=medicine_labels, hovertemplate='%{customdata}', xgap=0.1, ygap=0.1)
    # fig.write_html(os.path.join(args.output, "processing_count.html"))

    fig_times = px.imshow(processing_times_matrix, title="Processing time for each machine and interface in the solution",
                    color_continuous_scale="Viridis", text_auto=True)
    fig_times.update_layout(
        xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, mirror=True, showline=True, linecolor='lightgray'),
        yaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, mirror=True, showline=True, linecolor='lightgray')
    )
    fig_times.update_traces(customdata=medicine_labels, hovertemplate='%{customdata}', 
                            text=processing_times_text, texttemplate="%{text}",
                            xgap=0.1, ygap=0.1)
    file_path = os.path.join(args.output, "processing_times.html")
    fig_times.write_html(file_path, include_plotlyjs='cdn', full_html=True)
    inject_highlight_script(file_path)


def inject_highlight_script(file_path):
    """ Injects script to highlight on click same dispensers in the plot """
    js_script = """
    <script>
        document.addEventListener('DOMContentLoaded', function () {
            var plot = document.getElementsByClassName('plotly-graph-div')[0];

            var z_curr = plot.data[0].z;
            var customData = plot.data[0].customdata;
            console.log(customData);

            plot.on('plotly_click', function (data) {
                var clickedCustomData = data.points[0].customdata;
                if (clickedCustomData === 'empty' || clickedCustomData === 'blocked') {
                    return
                }
                var allCustomData = data.points[0].data.customdata // [["a", "b"], ["c", "d"]]
                const z = data.points[0].fullData.z
                let newText = Array.from(Array(z.length), () => Array(z[0].length).fill(''));

                for (let i = 0; i < allCustomData.length; i++) {
                    for (let j = 0; j < allCustomData[i].length; j++) {
                        if (allCustomData[i][j] === 'interface') {
                            newText[i][j] = 'I'
                        } else if (allCustomData[i][j] === 'blocked' || allCustomData[i][j] === 'empty') {
                            newText[i][j] = allCustomData[i][j]
                        } 
                        // Clicked = multiple, other multiple
                        else if (clickedCustomData.includes(',') && allCustomData[i][j].includes(',')) {
                            let splittedClicked = allCustomData[i][j].split(',')
                            let splittedOther = clickedCustomData.split(',')
                            let intersection = splittedClicked.filter(x => splittedOther.includes(x))
                            if (intersection.length > 0) {
                                newText[i][j] = String(z[i][j])
                            } else {
                                newText[i][j] = ' '
                            }
                        }
                        // Clicked = multiple, other single
                        else if (clickedCustomData.includes(',') && !allCustomData[i][j].includes(',')) {
                            let splittedClicked = clickedCustomData.split(',')
                            if (splittedClicked.includes(allCustomData[i][j])) {
                                newText[i][j] = String(z[i][j])
                            } else {
                                newText[i][j] = ' '
                            }
                        }
                        // Clicked = single, other multiple
                        else if (!clickedCustomData.includes(',') && allCustomData[i][j].includes(',')) {
                            let splittedOther = allCustomData[i][j].split(',')
                            if (splittedOther.includes(clickedCustomData)) {
                                newText[i][j] = String(z[i][j])
                            } else {
                                newText[i][j] = ' '
                            }
                        }
                        // Clicked = single, other single 
                        else if (allCustomData[i][j] === clickedCustomData) {
                            newText[i][j] = String(z[i][j])
                        } 
                        else {
                            newText[i][j] = ' '
                        }
                    }
                }

                Plotly.restyle(plot, {
                    'text': [newText],
                });
            });

            plot.on('plotly_doubleclick', function () {
                let newText = Array.from(Array(z_curr._inputArray.length), () => Array(z_curr._inputArray[0].length).fill(''));
                for (let i = 0; i < z_curr._inputArray.length; i++) {
                    for (let j = 0; j < z_curr._inputArray[i].length; j++) {
                        if (customData[i][j] === 'empty' || customData[i][j] === 'blocked') {
                            newText[i][j] = customData[i][j]
                        } else if (customData[i][j] === 'interface') {
                            newText[i][j] = 'I: ' + String(z_curr._inputArray[i][j])
                        } else {
                            newText[i][j] = String(z_curr._inputArray[i][j])
                        }
                    }
                }
                setTimeout(function (){
                    Plotly.restyle(plot, {
                    'text': [newText],
                });
                }, 30);
            });
        });
    </script>
    """

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    content = content.replace("</body>", js_script + "\n</body>")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def load_layout(args):
    layout = json.load(open(args.layout, "r"))
    layout["interface_locations"] = [tuple(loc) for loc in layout["interface_locations"]]
    layout["unavailable_locations"] = [tuple(loc) for loc in layout["unavailable_locations"]]
    return layout

def load_drug_dosing():
    drug_dosing = {}
    with open("drugs_dosing.csv", "r") as drugs_dosing_file:
        reader = csv.reader(drugs_dosing_file, delimiter=";")
        next(reader)
        for line in reader:
            drug_dosing[line[0]] = np.mean([int(line[-2]), int(line[-1])])
    return drug_dosing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimizes dispenser placement.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-p", "--packing", help="packer json file", type=str)
    parser.add_argument("-l", "--layout", help="json file with layout info: n, m, unavailable locations, interface locations", type=str) 
    parser.add_argument("-i", "--interfaces", help="number of interface locations", type=int, default=2)
    parser.add_argument("-o", "--output", help="output directory for placement", type=str, default="")
    parser.add_argument("--evals", help="maximum number of evaluations", type=int, default=30000)
    parser.add_argument("--pop-size", help="size of population", type=int, default=100)
    parser.add_argument("--episodes", help="number of simulation episodes per patient", type=int, default=5)
    parser.add_argument("--processes", help="number of processes for optimization", type=int, default=10)
    parser.add_argument("--save-figs", help="save option for placement figure",  dest='figs', default=False)
    parser.add_argument("--save-checkpoints", help="save intermediate checkpoints along the solution process", dest='checkpoints', default=False)

    args = parser.parse_args()
    layout: Layout = load_layout(args)

    drug_dosing = load_drug_dosing() # TODO would make more sense to store it in packer json file, and load it from there

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

    placement, res = compute_placement(args, layout, patient_list, sorted_names, drug_packing, drug_dosing)

    # Plot init population
    # plot_init_pop(args, layout, drug_packing, res)

    save_processing_times_plot(placement, res.algorithm.callback.data["processing_count"][-1], res.algorithm.callback.data["processing_time"][-1], args)

    if args.checkpoints:
        solutions = res.algorithm.callback.data["solution"]
        best_obj = res.algorithm.callback.data["best"]
        mean_obj = res.algorithm.callback.data["mean"]

        first_solution = format_placement(solutions[0], layout, n_tiles, args.interfaces)
        save_placement(first_solution, [best_obj[0]], [mean_obj[0]], args, checkpoint=0)
        prev_obj = best_obj[0]

        for sol_id in range(1, len(best_obj)):
            if best_obj[sol_id] < prev_obj:
                placement = format_placement(solutions[sol_id], layout, n_tiles, args.interfaces)
                save_placement(placement, best_obj[0:(sol_id+1)], mean_obj[0:(sol_id+1)], args, checkpoint=sol_id)
                prev_obj = best_obj[sol_id]
    else:
        save_placement(placement, res.algorithm.callback.data["best"], res.algorithm.callback.data["mean"], args)
