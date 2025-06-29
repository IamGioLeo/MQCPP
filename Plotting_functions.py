import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def plot_solution(graph, solution, graph_name=None, gamma=None):
    cmap = plt.get_cmap('tab20')
    color_map = [cmap(i) for i in range(20)]
    node_color_dict = {}
    edge_color_dict = {}
    legend_patches = []

    for index, subgraph in enumerate(solution):
        color = color_map[index % len(color_map)]
        if graph_name and gamma:
            legend_label = f"{gamma}-Clique {index + 1}"
        else:
            legend_label = f"Gamma-Clique {index + 1}"
        legend_patches.append(mpatches.Patch(color=color, label=legend_label))
        for node in subgraph.nodes():
            node_color_dict[node] = color
        for edge in subgraph.edges():
            edge_color_dict[tuple(sorted(edge))] = color

    nodes_colors = [node_color_dict.get(node, 'gray') for node in graph.nodes()]
    edges_colors = [edge_color_dict.get(tuple(sorted(edge)), 'gray') for edge in graph.edges()]

    pos = nx.spring_layout(graph, seed=42)

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title(
        f"Grafo {graph_name} con gamma = {gamma}" if (graph_name and gamma) else "Grafo")
    nx.draw(graph, pos,
            node_color=nodes_colors,
            edge_color=edges_colors,
            with_labels=True,
            node_size=400,
            font_size=8,
            width=2)

    if graph_name and gamma:
        plt.title(f"Grafo {graph_name} con gamma = {gamma}")
    else:
        plt.title("Grafo principale con sottografi evidenziati")

    plt.legend(handles=legend_patches, title="Sottografi", loc="best")

    plt.show()


def plot_gurobi_solution(graph, x, UB, gamma=None, graph_name=None):
    cmap = plt.get_cmap('tab20')
    color_map = [cmap(i) for i in range(20)]
    node_color_dict = {}
    edge_color_dict = {}
    legend_patches = []

    n = graph.number_of_nodes()
    pos = nx.spring_layout(graph, seed=42)

    clusters = {i: [] for i in range(UB)}
    for v in range(1, n + 1):
        for i in range(UB):
            try:
                if x[v, i].X > 0.5:
                    clusters[i].append(v)
                    break
            except Exception:
                print(f"\033[93mErrore: non è possibile accedere alla variabile X di gurobi\033[0m")
                return

    clusters = {i: nodes for i, nodes in clusters.items() if nodes}

    for idx, (cluster_id, nodes) in enumerate(clusters.items()):
        color = color_map[idx % len(color_map)]
        if graph_name and gamma:
            legend_label = f"{gamma}-Cluster {idx + 1}"
        else:
            legend_label = f"Cluster {idx + 1}"
        legend_patches.append(mpatches.Patch(color=color, label=legend_label))

        for node in nodes:
            node_color_dict[node] = color

        for u in nodes:
            for v in nodes:
                if u < v and graph.has_edge(u, v):
                    edge_color_dict[(u, v)] = color

    nodes_colors = [node_color_dict.get(node, 'gray') for node in graph.nodes()]
    edges_colors = [edge_color_dict.get(tuple(sorted(edge)), 'gray') for edge in graph.edges()]

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title(
        f"Grafo {graph_name} con gamma = {gamma}" if (graph_name and gamma) else "Grafo")

    nx.draw(graph, pos,
            node_color=nodes_colors,
            edge_color=edges_colors,
            with_labels=True,
            node_size=400,
            font_size=8,
            width=2)

    if graph_name and gamma:
        plt.title(f"Grafo {graph_name} con gamma = {gamma}")
    else:
        plt.title("Grafo con cluster colorati")

    plt.legend(handles=legend_patches, title="Cluster", loc="best")
    plt.show()
