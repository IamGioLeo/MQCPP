import gurobipy as gb
from gurobipy import GRB
from MQCPP import (MQCPP_solver)


def gurobi(graph, gamma):
    n = graph.number_of_nodes()
    initial_solution = MQCPP_solver(graph, gamma).generate_initial_solution()
    UB = max(initial_solution) + 1
    MQCPP = gb.Model()
    MQCPP.setParam("OutputFlag", 0)

    y = MQCPP.addVars([i for i in range(UB)], vtype=gb.GRB.BINARY)
    x = MQCPP.addVars([(v, i) for v in range(1, n + 1) for i in range(UB)], vtype=gb.GRB.BINARY)
    w = MQCPP.addVars([(u, v, i) for u in range(1, n + 1) for v in range(1, n + 1) if u < v for i in range(UB)], vtype=gb.GRB.BINARY)

    MQCPP.setObjective(gb.quicksum(y[i] for i in range(UB)), sense=gb.GRB.MINIMIZE)

    MQCPP.addConstrs(gb.quicksum(x[v, i] for i in range(UB)) == 1 for v in range(1, n + 1))
    MQCPP.addConstrs(x[v,i] <= y[i] for v in range(1, n + 1) for i in range(UB))
    MQCPP.addConstrs(x[u,i]+ x[v,i] <= w[u,v,i] + 1 for u in range(1, n + 1) for v in range(1, n + 1) if u < v for i in range(UB))
    MQCPP.addConstrs(w[u,v,i] <= x[u,i] for u in range(1, n + 1) for v in range(1, n + 1) if u < v for i in range(UB))
    MQCPP.addConstrs(w[u,v,i] <= x[v,i] for u in range(1, n + 1) for v in range(1, n + 1) if u < v for i in range(UB))
    MQCPP.addConstrs(y[i] >= y [i + 1] for i in range(UB-1))
    MQCPP.addConstrs(gb.quicksum(w[u,v,i] for u in range(1, n + 1) for v in range(1, n + 1) if (u < v and graph.has_edge(u,v))) >= gamma * gb.quicksum(w[u,v,i] for u in range(1, n + 1) for v in range(1, n + 1) if (u < v)) for i in range(UB))

    MQCPP.optimize()

    return MQCPP.objVal, x, UB, MQCPP
'''


def gurobi(graph, gamma):
    nodes = list(graph.nodes())
    initial_solution = MQCPP_solver(graph, gamma).generate_initial_solution()
    UB = max(initial_solution) + 1
    print(f"valore di UB per questa istanza{UB}")
    MQCPP = gb.Model()
    edges = {(u, v) for u, v in graph.edges() if u < v}
    print(edges)

    y = MQCPP.addVars([i for i in range(UB)], vtype=gb.GRB.BINARY)
    x = MQCPP.addVars([(v, i) for v in nodes for i in range(UB)], vtype=gb.GRB.BINARY)
    w = MQCPP.addVars([(u, v, i) for u in nodes for v in nodes if (u, v) in edges for i in range(UB)],
                      vtype=gb.GRB.BINARY)

    MQCPP.setObjective(gb.quicksum(y[i] for i in range(UB)), sense=gb.GRB.MINIMIZE)

    MQCPP.addConstrs(gb.quicksum(x[v, i] for i in range(UB)) == 1 for v in nodes)
    MQCPP.addConstrs(x[v, i] <= y[i] for v in nodes for i in range(UB))
    MQCPP.addConstrs(
        x[u, i] + x[v, i] <= w[u, v, i] + 1 for u in nodes for v in nodes if (u, v) in edges for i in range(UB))
    MQCPP.addConstrs(w[u, v, i] <= x[u, i] for u in nodes for v in nodes if (u, v) in edges for i in range(UB))
    MQCPP.addConstrs(w[u, v, i] <= x[v, i] for u in nodes for v in nodes if (u, v) in edges for i in range(UB))
    MQCPP.addConstrs(y[i] >= y[i + 1] for i in range(UB - 1))
    #il vincolo espresso così è quadratico, come lo hanno espresso loro no ma si generano molt più variabili perché serve una w per ogni possibile arco e non solo per quelli esistenti
    for i in range(UB):
        MQCPP.addConstr(
            gb.quicksum(w[u, v, i] for u, v in edges)
            >= gamma * (
                    gb.quicksum(x[v, i] for v in nodes) * (gb.quicksum(x[v, i] for v in nodes) - 1) / 2
            )
        )
    MQCPP.optimize()
    return MQCPP.objVal
'''