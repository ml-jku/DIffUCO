import numpy as np
import gurobipy as g


def solve_iGraph(i_graph, time_limit=float("inf"), solution_limit=None, verbose=0):
    nodes = i_graph.vs['external_field']
    edges = i_graph.es['couplings']
    senders = i_graph['senders']
    receivers = i_graph['receivers']
    N = len(nodes)

    m = g.Model("mip1")
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)

    obj1 = g.quicksum((2*var_dict[int(s)]-1) * weight * (2*var_dict[int(r)]-1) for s, r, weight in zip(senders, receivers, edges))
    obj2 = g.quicksum((2*var_dict[int(n)]-1) * nodes[n] for n in range(N))

    m.setObjective(obj1+obj2, g.GRB.MINIMIZE)
    m.optimize()

    return m, m.ObjVal, np.array([var_dict[key].x for key in var_dict]), m.Runtime