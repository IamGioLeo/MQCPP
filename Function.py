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


def function(instances_file_paths, gammas, cutoff_times, plot_flag: bool = True, gurobi_flag: bool = False,
             csv_path: str = '', alpha: int = 10000, beta: int = 5000):
    for file_path in instances_file_paths:
        graph_name = file_path.split("/")[-1]
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
            graph = nx.parse_edgelist(edge_lines, nodetype=int)
            tabu_tenure_upperbound = int(floor(graph.number_of_nodes() * 0.2))
            if graph.number_of_nodes() <= 118:
                for gamma in gammas:
                    print(f"\nInstance: [Graph: {graph_name}, Gamma: {gamma}]")
                    print(f"Number of nodes:{graph.number_of_nodes()}, number of edges: {graph.number_of_edges()}\n")
                    print("ILS result:")
                    start = datetime.datetime.now()
                    mqcpp_ILS = MQCPP_solver(graph, gamma)
                    best_solution = mqcpp_ILS.iterated_local_search(cutoff_times[0], alpha, beta,
                                                                    tabu_tenure_upperbound)

                    total_time = (datetime.datetime.now() - start).total_seconds()
                    best_solution_time = mqcpp_ILS.best_solution_time
                    subgraphs = mqcpp_ILS.solution_to_subgraphs(best_solution)
                    number_of_cliques = len(np.unique(best_solution))
                    print(f"Total time: {total_time}")
                    print(f"Time needed for best solution: {best_solution_time}")
                    print(f"Best solution: {number_of_cliques}\n ")
                    if plot_flag:
                        plot_solution(graph, subgraphs, graph_name, gamma)

                    if csv_path:
                        result = [{
                            "Graph": graph_name,
                            "Dimension": 'small',
                            "#nodes": graph.number_of_nodes(),
                            "#edges": graph.number_of_edges(),
                            "Gamma": gamma,
                            "Time": total_time,
                            "Initial_solution_time": mqcpp_ILS.initial_solution_time,
                            "Best_solution_time": best_solution_time,
                            "Best_solution_size": number_of_cliques,
                            "Solver": 'ILS'
                        }]
                        results_ILS_path = csv_path + "_ILS.csv"
                        intestazioni = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time", "Initial_solution_time","Best_solution_time", "Best_solution_size",
                                        "Solver"]
                        file_exists = os.path.isfile(results_ILS_path)
                        with open(results_ILS_path, mode="a", newline="") as file_csv:
                            writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
                            if not file_exists:
                                writer.writeheader()
                            writer.writerows(result)
                        print(f"Risultati aggiunti in {results_ILS_path}")

                    if gurobi_flag:
                        print(f"gurobi result:")
                        try:
                            starting_time = datetime.datetime.now()
                            best_solution, x, UB, MQCPP = gurobi(graph, gamma)
                            total_time = (datetime.datetime.now() - starting_time).total_seconds()
                            if plot_flag:
                                plot_gurobi_solution(graph, x, UB, gamma, graph_name)
                            print(f"Time needed: {total_time}")
                            print(f"Best solution: {best_solution}\n ")
                            if csv_path:
                                result = [{
                                    "Graph": graph_name,
                                    "Dimension": 'small',
                                    "#nodes": graph.number_of_nodes(),
                                    "#edges": graph.number_of_edges(),
                                    "Gamma": gamma,
                                    "Time": total_time,
                                    "Initial_solution_time": total_time,
                                    "Best_solution_time": None,
                                    "Best_solution_size": best_solution,
                                    "Solver": 'gurobi'
                                }]
                                results_gurobi_path = csv_path + "_gurobi.csv"
                                intestazioni = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time",
                                                "Initial_solution_time", "Best_solution_time", "Best_solution_size",
                                                "Solver"]
                                file_exists = os.path.isfile(results_gurobi_path)
                                with open(results_gurobi_path, mode="a", newline="") as file_csv:
                                    writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerows(result)
                                print(f"Risultati aggiunti in {results_gurobi_path}")
                        except gb.GurobiError as e:
                            print(f"\033[93mErrore Gurobi: {e}\033[0m")
                            print(
                                "\033[93mIl modello è troppo grande per la licenza limitata. Si passa al prossimo grafo.\033[0m")
                            if csv_path:
                                result = [{
                                    "Graph": graph_name,
                                    "Dimension": 'small',
                                    "#nodes": graph.number_of_nodes(),
                                    "#edges": graph.number_of_edges(),
                                    "Gamma": gamma,
                                    "Time": total_time,
                                    "Initial_solution_time": None,
                                    "Best_solution_time": None,
                                    "Best_solution_size": None,
                                    "Solver": 'gurobi'
                                }]
                                results_gurobi_path = csv_path + "_gurobi.csv"
                                intestazioni = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time",
                                                "Initial_solution_time", "Best_solution_time", "Best_solution_size",
                                                "Solver"]
                                file_exists = os.path.isfile(results_gurobi_path)
                                with open(results_gurobi_path, mode="a", newline="") as file_csv:
                                    writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerows(result)
                                print(f"Risultati aggiunti in {results_gurobi_path}")
                            continue


            elif graph.number_of_nodes() > 118:
                for gamma in gammas:
                    print(f"\nInstance: [Graph: {graph_name}, Gamma: {gamma}]")
                    print(f"Number of nodes:{graph.number_of_nodes()}, number of edges: {graph.number_of_edges()}\n")
                    print("ILS result:")
                    start = datetime.datetime.now()
                    mqcpp_ILS = MQCPP_solver(graph, gamma)
                    best_solution = mqcpp_ILS.iterated_local_search(cutoff_times[1], alpha, beta,
                                                                    tabu_tenure_upperbound)
                    total_time = (datetime.datetime.now() - start).total_seconds()
                    best_solution_time = mqcpp_ILS.best_solution_time
                    number_of_cliques = len(np.unique(best_solution))
                    print(f"Total time: {total_time}")
                    print(f"Time needed for best solution: {best_solution_time}")
                    print(f"Best solution: {number_of_cliques}\n ")
                    if csv_path:
                        result = [{
                            "Graph": graph_name,
                            "Dimension": 'large',
                            "#nodes": graph.number_of_nodes(),
                            "#edges": graph.number_of_edges(),
                            "Gamma": gamma,
                            "Time": total_time,
                            "Initial_solution_time": mqcpp_ILS.initial_solution_time,
                            "Best_solution_time": best_solution_time,
                            "Best_solution_size": number_of_cliques,
                            "Solver": 'ILS'
                        }]
                        results_ILS_path = csv_path + "_ILS.csv"
                        intestazioni = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time",
                                        "Initial_solution_time", "Best_solution_time", "Best_solution_size",
                                        "Solver"]
                        file_exists = os.path.isfile(results_ILS_path)
                        with open(results_ILS_path, mode="a", newline="") as file_csv:
                            writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
                            if not file_exists:
                                writer.writeheader()
                            writer.writerows(result)
                        print(f"Risultati aggiunti in {results_ILS_path}")

                    if gurobi_flag:
                        print(f"gurobi result:")
                        try:
                            starting_time = datetime.datetime.now()
                            best_solution, _, _, _ = gurobi(graph, gamma)
                            total_time = (datetime.datetime.now() - starting_time).total_seconds()
                            print(f"Time needed: {total_time}")
                            print(f"Best solution: {best_solution} \n ")
                            if csv_path:
                                result = [{
                                    "Graph": graph_name,
                                    "Dimension": 'large',
                                    "#nodes": graph.number_of_nodes(),
                                    "#edges": graph.number_of_edges(),
                                    "Gamma": gamma,
                                    "Time": total_time,
                                    "Initial_solution_time": None,
                                    "Best_solution_time": total_time,
                                    "Best_solution_size": best_solution,
                                    "Solver": 'gurobi'
                                }]
                                results_gurobi_path = csv_path + "_gurobi.csv"
                                intestazioni = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time",
                                                "Initial_solution_time", "Best_solution_time", "Best_solution_size",
                                                "Solver"]
                                file_exists = os.path.isfile(results_gurobi_path)
                                with open(results_gurobi_path, mode="a", newline="") as file_csv:
                                    writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerows(result)
                                print(f"Risultati aggiunti in {results_gurobi_path}")
                        except gb.GurobiError as e:
                            print(f"\033[93mErrore Gurobi: {e}\033[0m")
                            print(
                                "\033[93mIl modello è troppo grande per la licenza limitata. Si passa al prossimo grafo.\033[0m")
                            if csv_path:
                                result = [{
                                    "Graph": graph_name,
                                    "Dimension": 'large',
                                    "#nodes": graph.number_of_nodes(),
                                    "#edges": graph.number_of_edges(),
                                    "Gamma": gamma,
                                    "Time": 'none',
                                    "Best_solution_size": 'none',
                                    "Solver": 'gurobi'
                                }]
                                results_gurobi_path = csv_path + "_gurobi.csv"
                                intestazioni = ["Graph", "Dimension", "#nodes", "#edges", "Gamma", "Time",
                                                "Initial_solution_time", "Best_solution_time", "Best_solution_size",
                                                "Solver"]
                                file_exists = os.path.isfile(results_gurobi_path)
                                with open(results_gurobi_path, mode="a", newline="") as file_csv:
                                    writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerows(result)
                                print(f"Risultati aggiunti in {results_gurobi_path}")
                            continue