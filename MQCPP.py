import datetime
import itertools
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np


def objective_function(solution):
    return int(solution.max()) + 1


def sort_based_degree(graph):
    return sorted(graph.nodes, key=lambda x: graph.degree(x), reverse=True)


def update_M_union(M, i, j):
    new_col = M[:, i] + M[:, j]
    M_new = np.hstack([M, new_col.reshape(-1, 1)])
    return M_new


def update_M_reallocation(graph, M, node, i, j, indexes):
    node_idx = indexes[node]

    for neighbor in graph.neighbors(node):
        v_idx = indexes[neighbor]
        if v_idx != node_idx:
            M[v_idx, i] -= 1
            M[v_idx, j] += 1

    return M


def update_M_allocation(graph, M, node, i, indexes):
    node_idx = indexes[node]

    for neighbor in graph.neighbors(node):
        v_idx = indexes[neighbor]
        if v_idx != node_idx:
            M[v_idx, i] += 1
    return M


def update_M_addition(graph, M, node, indexes):
    new_col = np.zeros((M.shape[0],), dtype=int)
    for neighbor in graph.neighbors(node):
        new_col[indexes[neighbor]] += 1

    M = np.hstack([M, new_col.reshape(-1, 1)])
    return M


def update_M_removal(M, i):
    keep_cols = [k for k in range(M.shape[1]) if k != i]
    M = M[:, keep_cols]
    return M


def update_tabu_list(tabu_list):
    tabu_list[:] = [(v, t - 1) for v, t in tabu_list if t > 1]


class MQCPP_solver:
    def __init__(self, graph, gamma):
        self.graph = graph
        self.gamma = gamma
        self.n = graph.number_of_nodes()
        self.z = np.zeros(self.n + 1)
        for i in range(self.n + 1):
            self.z[i] = (i * (i - 1) / 2) * self.gamma
        self.M = np.empty((self.n, 0))
        self.tabu_list = []
        self.tabu_tenure_upper_bound = -1
        self.node_to_index = {u: r for r, u in enumerate(graph.nodes())}
        self.index_to_node = {i: u for u, i in self.node_to_index.items()}
        self.gamma_clique_edges_and_nodes = []
        self.best_solution_time = None
        self.initial_solution_time = None

    def nodes_to_clique_edge_count(self, member_indices, clique_index):
        count = 0
        for i in member_indices:
            count += self.M[i, clique_index]
        return count

    def update_solution(self, solution, gamma_clique_to_check):
        if gamma_clique_to_check in solution:
            return solution

        self.M = update_M_removal(self.M, gamma_clique_to_check)
        del self.gamma_clique_edges_and_nodes[gamma_clique_to_check]
        solution = np.where(solution > gamma_clique_to_check, solution - 1, solution)
        return solution

    def verify_reallocate(self, node, from_clique_id, to_clique_id):
        edge_count_from = self.gamma_clique_edges_and_nodes[from_clique_id][0] - self.M[self.node_to_index[node]][
            from_clique_id]

        edge_count_to = self.gamma_clique_edges_and_nodes[to_clique_id][0] + self.M[self.node_to_index[node]][
            to_clique_id]

        valid_from = edge_count_from >= self.z[self.gamma_clique_edges_and_nodes[from_clique_id][1] - 1]
        valid_to = edge_count_to >= self.z[self.gamma_clique_edges_and_nodes[to_clique_id][1] + 1]

        return valid_from and valid_to

    def verify_allocation(self, node, clique_id):
        edge_count = self.gamma_clique_edges_and_nodes[clique_id][0] + self.M[self.node_to_index[node]][clique_id]

        return edge_count >= self.z[self.gamma_clique_edges_and_nodes[clique_id][1] + 1]

    def identify_gamma_clique_to_remove(self, solution):
        gamma_clique_identifiers = list(range(len(self.gamma_clique_edges_and_nodes)))
        if not gamma_clique_identifiers:
            return -2

        unallocable_counts = {gamma_clique_id: 0 for gamma_clique_id in gamma_clique_identifiers}

        for node in self.graph.nodes():
            node_idx = self.node_to_index[node]
            current_clique = solution[node_idx]

            if current_clique == -1:
                continue

            allocable_elsewhere = False
            for other_clique in gamma_clique_identifiers:
                if other_clique == current_clique:
                    continue
                if self.verify_allocation(node, other_clique):
                    allocable_elsewhere = True
                    break

            if not allocable_elsewhere:
                unallocable_counts[current_clique] += 1

        min_count = min(unallocable_counts.values())
        candidates = [c for c, count in unallocable_counts.items() if count == min_count]

        return random.choice(candidates)

    def identify_best_reallocation(self, solution, tabu_list):
        best_gain = -float('inf')
        best_moves = []

        gamma_clique_identifiers = list(range(len(self.gamma_clique_edges_and_nodes)))

        for node_idx in range(len(solution)):
            node = self.index_to_node[node_idx]
            if any(tabu_node == node for tabu_node, _ in tabu_list):
                continue

            from_clique = solution[node_idx]
            if from_clique == -1:
                continue

            for to_clique in gamma_clique_identifiers:
                if to_clique == from_clique:
                    continue

                if self.verify_reallocate(node, from_clique, to_clique):
                    gain = self.M[node_idx, to_clique] - self.M[node_idx, from_clique]
                    if gain > best_gain:
                        best_gain = gain
                        best_moves = [(node, from_clique, to_clique)]
                    elif gain == best_gain:
                        best_moves.append((node, from_clique, to_clique))

        if not best_moves:
            return None, None, None

        return random.choice(best_moves)

    def identify_max_intersection_gamma_cliques(self, solution):
        best_i, best_j = -1, -1
        max_intersection = -1

        gamma_clique_identifiers = list(range(len(self.gamma_clique_edges_and_nodes)))

        for idx_i, ci in enumerate(gamma_clique_identifiers):
            for cj in gamma_clique_identifiers[idx_i + 1:]:
                indices_i = np.where(solution == ci)[0]

                inter_edges = self.nodes_to_clique_edge_count(indices_i, cj)

                if inter_edges > max_intersection:
                    max_intersection = inter_edges
                    best_i, best_j = ci, cj
        return best_i, best_j

    def reallocate(self, solution, node, from_clique, to_clique):
        node_idx = self.node_to_index[node]
        solution[node_idx] = to_clique

        tenure = random.randint(int(np.ceil(0.1 * self.n)), int(self.tabu_tenure_upper_bound))
        self.tabu_list.append((node, tenure))

        self.M = update_M_reallocation(self.graph, self.M, node, from_clique, to_clique, self.node_to_index)
        self.gamma_clique_edges_and_nodes[from_clique][0] -= self.M[node_idx][from_clique]
        self.gamma_clique_edges_and_nodes[from_clique][1] -= 1
        self.gamma_clique_edges_and_nodes[to_clique][0] += self.M[node_idx][to_clique]
        self.gamma_clique_edges_and_nodes[to_clique][1] += 1
        solution = self.update_solution(solution, from_clique)
        return solution

    def gamma_clique_merge(self, solution):
        merged = True

        while merged:
            merged = False
            gamma_clique_identifiers = list(range(len(self.gamma_clique_edges_and_nodes)))
            best_pair = None
            best_density = self.gamma

            for ci, cj in itertools.combinations(gamma_clique_identifiers, 2):
                indices_i = np.where(solution == ci)[0]
                num_nodes = len(indices_i) + self.gamma_clique_edges_and_nodes[cj][1]

                if num_nodes <= 1:
                    continue

                edge_count = self.gamma_clique_edges_and_nodes[ci][0] + self.gamma_clique_edges_and_nodes[cj][
                    0] + self.nodes_to_clique_edge_count(indices_i, cj)

                density = (2 * edge_count) / (num_nodes * (num_nodes - 1))

                if density >= best_density:
                    best_density = density
                    best_pair = (ci, cj)

            if best_pair:
                ci, cj = best_pair
                new_clique_id = int(solution.max()) + 1
                solution[np.isin(solution, [ci, cj])] = new_clique_id
                merged = True

                self.M = update_M_union(self.M, ci, cj)
                self.gamma_clique_edges_and_nodes.append([(self.gamma_clique_edges_and_nodes[ci][0] +
                                                           self.gamma_clique_edges_and_nodes[cj][
                                                               0] + self.nodes_to_clique_edge_count(
                        np.where(solution == ci)[0], cj)), (self.gamma_clique_edges_and_nodes[ci][1] +
                                                            self.gamma_clique_edges_and_nodes[cj][1])])
                solution = self.update_solution(solution, max(ci, cj))
                solution = self.update_solution(solution, min(ci, cj))

        return solution

    def greedy_allocate(self, solution, node):
        node_idx = self.node_to_index[node]
        best_gamma_clique = None
        best_density = 0

        for gamma_clique_id in list(range(len(self.gamma_clique_edges_and_nodes))):

            clique_indices = np.where(solution == gamma_clique_id)[0]
            candidate_size = len(clique_indices) + 1

            edges_count = self.gamma_clique_edges_and_nodes[gamma_clique_id][0] + self.M[self.node_to_index[node]][gamma_clique_id]

            density = (2 * edges_count) / (candidate_size * (candidate_size - 1)) if candidate_size > 1 else 0

            if density > best_density:
                best_density = density
                best_gamma_clique = gamma_clique_id

        if best_gamma_clique is not None and best_density >= self.gamma:
            self.M = update_M_allocation(self.graph, self.M, node, best_gamma_clique, self.node_to_index)
            self.gamma_clique_edges_and_nodes[best_gamma_clique][0] += self.M[node_idx][best_gamma_clique]
            self.gamma_clique_edges_and_nodes[best_gamma_clique][1] += 1
            solution[node_idx] = best_gamma_clique
        else:
            self.M = update_M_addition(self.graph, self.M, node, self.node_to_index)
            self.gamma_clique_edges_and_nodes.append([0, 1])
            solution[node_idx] = int(solution.max()) + 1

        return solution

    def generate_initial_solution(self):
        nodes_list = sort_based_degree(self.graph)
        initial_solution = np.full(self.n, -1, dtype=int)

        while nodes_list:
            node = nodes_list.pop(0)
            initial_solution = self.greedy_allocate(initial_solution, node)
            initial_solution = self.gamma_clique_merge(initial_solution)

        return initial_solution

    def iterated_local_search(self, cutoff_time, alpha, beta, tabu_tenure_upper_bound):
        self.tabu_tenure_upper_bound = tabu_tenure_upper_bound

        initial_solution_starting_time = datetime.datetime.now()
        current_solution = self.generate_initial_solution()
        best_solution = current_solution.copy()
        self.initial_solution_time = (datetime.datetime.now() - initial_solution_starting_time).total_seconds()

        self.best_solution_time = self.initial_solution_time
        starting_time = datetime.datetime.now()
        while ((datetime.datetime.now() - starting_time).total_seconds() < cutoff_time):
            if objective_function(best_solution) == 1:
                break
            found_solution, current_solution = self.two_phase_local_search(current_solution, alpha, beta)
            if objective_function(found_solution) < objective_function(best_solution):
                best_solution = found_solution.copy()
                self.best_solution_time = (datetime.datetime.now() - initial_solution_starting_time).total_seconds()
            current_solution = self.greedy_based_perturbation(current_solution)

        return best_solution

    def two_phase_local_search(self, current_solution, alpha, beta):
        best_solution = current_solution.copy()

        while True:
            improve_flag = False
            temporary_solution = current_solution.copy()
            RRTS_solution = self.remove_repair_based_tabu_search(alpha, current_solution)
            current_solution = RRTS_solution

            if objective_function(RRTS_solution) >= objective_function(temporary_solution):
                MDTS_solution, current_solution = self.merge_driven_tabu_search(beta, current_solution)
                if objective_function(MDTS_solution) < objective_function(best_solution):
                    best_solution = MDTS_solution.copy()
                    improve_flag = True
            else:
                if objective_function(RRTS_solution) < objective_function(best_solution):
                    best_solution = RRTS_solution.copy()
                    improve_flag = True

            if not improve_flag:
                return best_solution, current_solution

    def remove_repair_based_tabu_search(self, search_depth, solution):
        best_solution = solution.copy()
        M_best_solution = np.copy(self.M)
        edges_per_clique_best_solution = deepcopy(self.gamma_clique_edges_and_nodes)

        while True:
            if np.all(solution == -1):
                break

            gamma_clique_to_remove = self.identify_gamma_clique_to_remove(solution)
            node_indices_to_reallocate = np.where(solution == gamma_clique_to_remove)[0]
            nodes_to_reallocate = [self.index_to_node[i] for i in node_indices_to_reallocate]
            nodes_to_reallocate.sort()

            solution[node_indices_to_reallocate] = -1
            solution = self.update_solution(solution, gamma_clique_to_remove)

            remaining_cliques = [c for c in np.unique(solution) if c != -1]
            nodes_to_reallocate, solution = self.reinsert(nodes_to_reallocate, solution,
                                                          remaining_cliques)

            i = 0
            while nodes_to_reallocate and i < search_depth:
                i += 1
                node, from_clique, to_clique = self.identify_best_reallocation(solution, self.tabu_list)
                if not node:
                    break
                old_max = objective_function(solution)
                solution = self.reallocate(solution, node, from_clique, to_clique)

                update_tabu_list(self.tabu_list)

                if (old_max == objective_function(solution)):
                    target_cliques = [from_clique, to_clique] if to_clique != from_clique else [to_clique]
                else:
                    target_cliques = [to_clique - 1]

                nodes_to_reallocate, solution = self.reinsert(nodes_to_reallocate, solution,
                                                              target_cliques)

            if (len(nodes_to_reallocate) + objective_function(solution)) < objective_function(best_solution):
                for node in nodes_to_reallocate:
                    idx = self.node_to_index[node]
                    self.M = update_M_addition(self.graph, self.M, node, self.node_to_index)
                    self.gamma_clique_edges_and_nodes.append([0, 1])
                    solution[idx] = int(solution.max()) + 1

                nodes_to_reallocate = None

            if not nodes_to_reallocate and objective_function(solution) < objective_function(best_solution):
                best_solution = solution.copy()
                M_best_solution = np.copy(self.M)
                edges_per_clique_best_solution = deepcopy(self.gamma_clique_edges_and_nodes)

            if nodes_to_reallocate:
                break

        self.M = M_best_solution
        self.gamma_clique_edges_and_nodes = edges_per_clique_best_solution
        return best_solution

    def merge_driven_tabu_search(self, search_depth, solution):
        no_improvement = 0
        best_solution = solution.copy()
        current_solution = solution

        while no_improvement < search_depth:
            no_improvement += 1

            node, from_clique, to_clique = self.identify_best_reallocation(current_solution, self.tabu_list)
            if node is None:
                break

            current_solution = self.reallocate(current_solution, node, from_clique, to_clique)
            update_tabu_list(self.tabu_list)

            current_solution = self.gamma_clique_merge(current_solution)

            if objective_function(current_solution) < objective_function(best_solution):
                best_solution = current_solution.copy()
                no_improvement = 0

        return best_solution, current_solution

    def greedy_based_perturbation(self, solution):
        gamma_clique_identifiers = list(range(len(self.gamma_clique_edges_and_nodes)))
        if len(gamma_clique_identifiers) < 2:
            return solution

        ci, cj = self.identify_max_intersection_gamma_cliques(solution)
        if ci == -1 or cj == -1:
            return solution

        nodes_i = np.where(solution == ci)[0]
        nodes_j = np.where(solution == cj)[0]
        affected_nodes = np.union1d(nodes_i, nodes_j)

        perturbed_solution = solution
        perturbed_solution[affected_nodes] = -1
        perturbed_solution = self.update_solution(perturbed_solution, int(max(ci, cj)))
        perturbed_solution = self.update_solution(perturbed_solution, int(min(ci, cj)))

        for node_idx in affected_nodes:
            node = self.index_to_node[node_idx]
            perturbed_solution = self.greedy_allocate(perturbed_solution, node)
            perturbed_solution = self.gamma_clique_merge(perturbed_solution)

        return perturbed_solution

    def reinsert(self, nodes_to_reallocate, solution, clusters_to_fill):
        if not nodes_to_reallocate or not clusters_to_fill:
            return nodes_to_reallocate, solution

        node = nodes_to_reallocate[0]
        node_idx = self.node_to_index[node]

        for gamma_clique_id in clusters_to_fill:
            if self.verify_allocation(node, gamma_clique_id):
                self.M = update_M_allocation(self.graph, self.M, node, gamma_clique_id, self.node_to_index)
                self.gamma_clique_edges_and_nodes[gamma_clique_id][0] += self.M[node_idx][gamma_clique_id]
                self.gamma_clique_edges_and_nodes[gamma_clique_id][1] += 1
                solution[node_idx] = gamma_clique_id
                return self.reinsert(nodes_to_reallocate[1:], solution, clusters_to_fill)

        return nodes_to_reallocate, solution

    def solution_to_subgraphs(self, solution):

        clusters = defaultdict(list)

        for idx, subgraph_id in enumerate(solution):
            node = self.index_to_node[idx]
            clusters[subgraph_id].append(node)

        subgraphs = [self.graph.subgraph(nodes).copy() for nodes in clusters.values()]
        return subgraphs
