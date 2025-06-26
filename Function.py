import datetime
import networkx as nx
from MQCPP import MQCPP_solver
from PlotSolution import plot_solution
from PlotSolution import plot_gurobi_solution
import numpy as np
from math import floor
import csv
import gurobipy as gb
from Gurobi import gurobi
import os


def _parse_graph_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        edge_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].lower() == 'e':
                parts = stripped.split()
                edge_lines.append(f"{parts[1]} {parts[2]}")
            elif stripped and stripped[0].isdigit():
                parts = stripped.split()
                edge_lines.append(f"{parts[0]} {parts[1]}")
        return nx.parse_edgelist(edge_lines, nodetype=int)


def _write_csv_results(csv_path, solver_name, result_data):
    if not csv_path:
        return

    results_path = f"{csv_path}_{solver_name}.csv"
    headers = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time",
               "Initial_solution_time", "Best_solution_time", "Best_solution_size", "Solver"]

    file_exists = os.path.isfile(results_path)
    with open(results_path, mode="a", newline="") as file_csv:
        writer = csv.DictWriter(file_csv, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerows([result_data])


def _create_result_data(graph_name, dimension, graph, gamma, total_time,
                        initial_time, best_time, best_solution, solver):
    return {
        "Graph": graph_name,
        "Dimension": dimension,
        "#nodes": graph.number_of_nodes(),
        "#edges": graph.number_of_edges(),
        "Gamma": gamma,
        "Time": total_time,
        "Initial_solution_time": initial_time,
        "Best_solution_time": best_time,
        "Best_solution_size": best_solution,
        "Solver": solver
    }


def _run_ils_solver(graph, gamma, cutoff_time, alpha, beta, tabu_tenure_upperbound):
    print("ILS result:")
    start = datetime.datetime.now()
    mqcpp_ILS = MQCPP_solver(graph, gamma)
    best_solution = mqcpp_ILS.iterated_local_search(cutoff_time, alpha, beta, tabu_tenure_upperbound)

    total_time = (datetime.datetime.now() - start).total_seconds()
    best_solution_time = mqcpp_ILS.best_solution_time
    number_of_cliques = len(np.unique(best_solution))

    print(f"Total time: {total_time}")
    print(f"Time needed for initial solution: {mqcpp_ILS.initial_solution_time}")
    print(f"Time needed for best solution: {best_solution_time}")
    print(f"Best solution: {number_of_cliques}\n ")

    return {
        'total_time': total_time,
        'initial_time': mqcpp_ILS.initial_solution_time,
        'best_time': best_solution_time,
        'best_solution': best_solution,
        'number_of_cliques': number_of_cliques,
        'subgraphs': mqcpp_ILS.solution_to_subgraphs(best_solution)
    }


def _run_gurobi_solver(graph, gamma):
    print("gurobi result:")
    starting_time = datetime.datetime.now()
    best_solution, x, UB, MQCPP = gurobi(graph, gamma)
    total_time = (datetime.datetime.now() - starting_time).total_seconds()

    print(f"Time needed: {total_time}")
    print(f"Best solution: {best_solution}\n ")

    return {
        'total_time': total_time,
        'best_solution': best_solution,
        'x': x,
        'UB': UB,
        'MQCPP': MQCPP

    }


def _handle_gurobi_error(graph_name, dimension, graph, gamma, csv_path):
    print(f"\033[93mErrore Gurobi: Il modello è troppo grande per la licenza limitata.\033[0m")
    print("\033[93mSi passa al prossimo grafo.\033[0m")

    if csv_path:
        error_result = _create_result_data(
            graph_name, dimension, graph, gamma,
            'none' if dimension == 'large' else None,
            None, None, 'none' if dimension == 'large' else None, 'gurobi'
        )
        _write_csv_results(csv_path, 'gurobi', error_result)


def _process_instance(graph, graph_name, gamma, cutoff_time, alpha, beta,
                      tabu_tenure_upperbound, dimension, plot_flag, gurobi_flag, csv_path):
    print(f"\nInstance: [Graph: {graph_name}, Gamma: {gamma}]")
    print(f"Number of nodes:{graph.number_of_nodes()}, number of edges: {graph.number_of_edges()}\n")

    ils_results = _run_ils_solver(graph, gamma, cutoff_time, alpha, beta, tabu_tenure_upperbound)

    if plot_flag:
        plot_solution(graph, ils_results['subgraphs'], graph_name, gamma)

    if csv_path:
        ils_result = _create_result_data(
            graph_name, dimension, graph, gamma,
            ils_results['total_time'], ils_results['initial_time'],
            ils_results['best_time'], ils_results['number_of_cliques'], 'ILS'
        )
        _write_csv_results(csv_path, 'ILS', ils_result)

    if gurobi_flag:
        try:
            gurobi_results = _run_gurobi_solver(graph, gamma)

            if plot_flag and dimension == 'small':
                plot_gurobi_solution(graph, gurobi_results['x'], gurobi_results['UB'], gamma, graph_name)

            if csv_path:
                gurobi_result = _create_result_data(
                    graph_name, dimension, graph, gamma,
                    gurobi_results['total_time'], None,
                    gurobi_results['total_time'], gurobi_results['best_solution'], 'gurobi'
                )
                _write_csv_results(csv_path, 'gurobi', gurobi_result)

        except gb.GurobiError as e:
            print(f"\033[93mErrore Gurobi: {e}\033[0m")
            _handle_gurobi_error(graph_name, dimension, graph, gamma, csv_path)


def function(instances_file_paths, gammas, cutoff_times, plot_flag: bool = True, gurobi_flag: bool = False,
             csv_path: str = '', alpha: int = 10000, beta: int = 5000):
    for file_path in instances_file_paths:
        graph_name = file_path.split("/")[-1]
        graph = _parse_graph_from_file(file_path)
        tabu_tenure_upperbound = int(floor(graph.number_of_nodes() * 0.2))
        if graph.number_of_nodes() <= 118:
            dimension = 'small'
            cutoff_time = cutoff_times[0]
        else:
            dimension = 'large'
            cutoff_time = cutoff_times[1]
        for gamma in gammas:
            _process_instance(graph, graph_name, gamma, cutoff_time, alpha, beta,
                              tabu_tenure_upperbound, dimension, plot_flag, gurobi_flag, csv_path)