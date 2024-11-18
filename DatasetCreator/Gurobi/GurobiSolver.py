import numpy as np
import gurobipy as g
import multiprocessing

def get_adjacency_list(edges):
    adjacency_list = {}
    for edge in edges:
        # Assuming the graph is undirected, so we add both directions
        if edge[0] not in adjacency_list:
            adjacency_list[edge[0]] = []
        adjacency_list[edge[0]].append(edge[1])

        if edge[1] not in adjacency_list:
            adjacency_list[edge[1]] = []
        adjacency_list[edge[1]].append(edge[0])
    return adjacency_list

def solveMDS_as_MIP(H_graph, time_limit = float("inf"), num_CPUs = None, thread_fraction = 0.5):

    num_nodes = H_graph.nodes.shape[0]
    m = g.Model("mip1")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    if(num_CPUs == None):
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(thread_fraction*multiprocessing.cpu_count()))
    else:
        m.setParam("Threads", int(num_CPUs))

    edge_list = [(s,r) for s,r in zip(H_graph.senders, H_graph.receivers)]
    adjacency_list = get_adjacency_list(edge_list)

    var_dict = m.addVars(num_nodes, vtype=g.GRB.BINARY)
    obj2 = g.quicksum( var_dict[int(n)]  for n in range(num_nodes))

    for node in adjacency_list:
        neighbors = adjacency_list[node]
        m.addConstr(var_dict[int(node)] + g.quicksum(var_dict[neighbor] for neighbor in neighbors) >= 1)

    m.setObjective(obj2, g.GRB.MINIMIZE)
    m.optimize()

    cover = []
    for v in var_dict:
        #print( vertexVars[v].X)
        if var_dict[v].X > 0.5:
            #print ("Vertex'," +str(v)+ 'is in the cover')
            cover.append(v)
    #print("MVC size",len(cover))
    return m, len(cover), np.array([var_dict[key].x for key in var_dict]), m.Runtime



def solveMVC_as_MIP(H_graph, time_limit = float("inf"), num_CPUs = None, thread_fraction = 0.5):

    num_nodes = H_graph.nodes.shape[0]
    m = g.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    if(num_CPUs == None):
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(thread_fraction*multiprocessing.cpu_count()))
    else:
        m.setParam("Threads", int(num_CPUs))

    edge_list = [(min([s,r]),max([s,r]))for s,r in zip(H_graph.senders, H_graph.receivers)]
    unique_edge_List = set(edge_list)

    var_dict = m.addVars(num_nodes, vtype=g.GRB.BINARY)
    obj2 = g.quicksum( var_dict[int(n)]  for n in range(num_nodes))

    for (s,r) in unique_edge_List:
        xs = var_dict[s]
        xr = var_dict[r]
        m.addConstr(xs + xr >= 1, name="e%d-%d" % (s, r))

    m.setObjective(obj2, g.GRB.MINIMIZE)
    m.optimize()

    cover = []
    for v in var_dict:
        #print( vertexVars[v].X)
        if var_dict[v].X > 0.5:
            #print ("Vertex'," +str(v)+ 'is in the cover')
            cover.append(v)
    #print("MVC size",len(cover))
    return m, len(cover), np.array([var_dict[key].x for key in var_dict]), m.Runtime


### TODO add version for weighted MIS
def solveMIS_as_MIP(H_graph,  time_limit = float("inf"), thread_fraction = 0.5, num_CPUs = None):

    num_nodes = H_graph.nodes.shape[0]
    m = g.Model("mip1")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    if(num_CPUs == None):
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(thread_fraction*multiprocessing.cpu_count()))
    else:
        m.setParam("Threads", int(num_CPUs))

    edge_list = [(int(min([s,r])),int(max([s,r]))) for s,r in zip(H_graph.senders, H_graph.receivers)]
    unique_edge_List = set(edge_list)

    var_dict = m.addVars(num_nodes, vtype=g.GRB.BINARY)
    obj2 = g.quicksum( -var_dict[int(n)]  for n in range(num_nodes))

    for (s,r) in unique_edge_List:
        xs = var_dict[s]
        xr = var_dict[r]
        m.addConstr(xs + xr <= 1, name="e%d-%d" % (s, r))

    m.setObjective(obj2, g.GRB.MINIMIZE)
    m.optimize()

    cover = []
    for v in var_dict:
        #print( vertexVars[v].X)
        if var_dict[v].X > 0.5:
            #print ("Vertex'," +str(v)+ 'is in the cover')
            cover.append(v)
    #print("MIS size",len(cover))
    return m, -len(cover), np.array([var_dict[key].x for key in var_dict]), m.Runtime

def solveWMIS_as_MIP_at_d(H_graph, ground_state, d,  time_limit = float("inf"), B = 1.1, A = 1):

    num_nodes = H_graph.nodes.shape[0]
    weights = H_graph.nodes
    m = g.Model("mip1")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    edge_list = [(min([s,r]),max([s,r]))for s,r in zip(H_graph.senders, H_graph.receivers)]
    unique_edge_List = set(edge_list)

    var_dict = m.addVars(num_nodes, vtype=g.GRB.BINARY)

    obj1 = g.quicksum( 0.5*B*var_dict[int(s)] * var_dict[int(r)] for s, r, in zip(H_graph.senders, H_graph.receivers))
    obj2 = g.quicksum( - A* weights[n,0]*var_dict[int(n)] * var_dict[int(n)]  for n in range(num_nodes))

    hamm_dist = g.quicksum((ground_state[n] -var_dict[n])**2  for n in range(num_nodes))
    m.addConstr(hamm_dist == d)

    m.setObjective(obj2+obj1, g.GRB.MINIMIZE)
    m.optimize()

    print(np.array([var_dict[key] for key in var_dict]))
    try:
        solution = np.array([var_dict[key].X for key in var_dict])
        sol_weight = -np.sum(solution*weights[:,0]) + sum([0.5*B*var_dict[int(s)].X * var_dict[int(r)].X for s, r, in zip(H_graph.senders, H_graph.receivers)])
    except:
        print("here")
    return m, sol_weight, solution, m.Runtime

def solveIsing_at_d(H_graph, ground_state, d,  time_limit = float("inf"), d_constraint = False):

    num_nodes = H_graph.nodes.shape[0]
    weights = H_graph.nodes
    edge_weights = H_graph.edges
    m = g.Model("mip1")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 0)

    var_dict = m.addVars(num_nodes, vtype=g.GRB.BINARY)

    obj1 = g.quicksum( 0.5*e[0]*(2*var_dict[int(s)]-1) * (2*var_dict[int(r)]-1) for s, r, e in zip(H_graph.senders, H_graph.receivers, edge_weights))
    obj2 = g.quicksum( weights[n,0]*(2*var_dict[int(n)]-1)  for n in range(num_nodes))

    if(d_constraint == True):
        hamm_dist = g.quicksum((ground_state[n] - var_dict[n]) ** 2 for n in range(num_nodes))
        m.addConstr(hamm_dist == d)

    m.setObjective(obj2+obj1, g.GRB.MINIMIZE)
    m.optimize()

    solution = np.array([var_dict[key].X for key in var_dict])
    sol_weight = m.ObjVal

    return m, sol_weight, solution, m.Runtime

def solveWMIS_as_MIP(H_graph,  time_limit = float("inf")):

    num_nodes = H_graph.nodes.shape[0]
    weights = H_graph.nodes
    m = g.Model("mip1")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    edge_list = [(min([s,r]),max([s,r]))for s,r in zip(H_graph.senders, H_graph.receivers)]
    unique_edge_List = set(edge_list)

    var_dict = m.addVars(num_nodes, vtype=g.GRB.BINARY)
    obj2 = g.quicksum( -weights[n,0]*var_dict[n]  for n in range(num_nodes))

    for (s,r) in unique_edge_List:
        xs = var_dict[s]
        xr = var_dict[r]
        m.addConstr(xs + xr <= 1, name="e%d-%d" % (s, r))

    m.setObjective(obj2, g.GRB.MINIMIZE)
    m.optimize()

    #print("MIS size",len(cover))
    solution = np.array([var_dict[key].x for key in var_dict])
    sol_weight = -np.sum(solution*weights[:,0])
    return m, sol_weight, solution, m.Runtime

def solveWMIS_QUBO(H_graph, time_limit=float("inf"), model_name = "miqp1", solution_limit=None, verbose=0, A = 1., B = 3., bnb = True, measure_time = False):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    weight = H_graph.nodes
    B = B
    A = A

    m = g.Model(model_name)
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    if (measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
        # m.params.mipfocus = 2
        # m.params.nodelimit = 1000000
        # m.params.branchdir = -1
    else:
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(0.75*multiprocessing.cpu_count()))
    if (bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)

    obj1 = B*g.quicksum( B*var_dict[int(s)] * var_dict[int(r)] for s, r, in zip(senders, receivers))
    obj2 = g.quicksum( - A* weight[n,0]*var_dict[int(n)] * var_dict[int(n)]  for n in range(N))

    m.setObjective(obj1 + obj2, g.GRB.MINIMIZE)
    m.optimize()

    print("MIS solution", m.ObjVal)
    return m, m.ObjVal, np.array([var_dict[key].x for key in var_dict]),  m.Runtime



def solveMaxClique(H_graph, time_limit=float("inf"), solution_limit=None, verbose=0, A = 1., B = 1.1, beta = 11., bnb = True, measure_time = False):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers

    m = g.Model("mip1")
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    if (measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
        # m.params.mipfocus = 2
        # m.params.nodelimit = 1000000
        # m.params.branchdir = -1
    if (bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)


    degree = np.zeros((H_graph.nodes.shape[0],1))
    ones_edges = np.ones_like(H_graph.edges)
    np.add.at(degree, H_graph.receivers, ones_edges)
    max_degree = int(np.max(degree)) + 1 ### TODO think about why this +1 is neccesary here in case of a fully conencted graph

    B = 1
    A = (max_degree + 2)*B
    C = B

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)
    var_dict_y = m.addVars(max_degree - 1, vtype=g.GRB.BINARY)

    sum_nodes = g.quicksum( var_dict[int(n)]  for n in range(N))
    sum_n_times_y = g.quicksum( (n + 2)*var_dict_y[int(n)]  for n in range(0, max_degree -1))
    HA1 = A*(1-g.quicksum( var_dict_y[int(n)]  for n in range(0, max_degree -1)))**2
    HA2 = A*(sum_n_times_y- sum_nodes)**2
    HA = HA1 + HA2

    HB11 = g.quicksum( 0.5*(n + 2)*var_dict_y[int(n)]  for n in range(0, max_degree - 1))
    HB12 = g.quicksum(((n + 2)*var_dict_y[int(n)]  for n in range(0, max_degree - 1)))
    HB1 = B*HB11 *(HB12 - 1)
    HB2 = B*g.quicksum( -1/2*var_dict[int(s)] * var_dict[int(r)] for s, r, in zip(senders, receivers))
    HB = HB1 + HB2
    HC = -C*g.quicksum( var_dict[int(n)]  for n in range(N))

    m.setObjective(HA + HB + HC, g.GRB.MINIMIZE)
    m.optimize()

    #print("MaxCliqueSize",m.ObjVal)
    y_sol = np.array([var_dict_y[key].x for key in var_dict_y])
    return m, m.ObjVal, np.array([var_dict[key].x for key in var_dict]),  m.Runtime

def solveMaxClique_log(H_graph, time_limit=float("inf"), solution_limit=None, verbose=0, A = 1., B = 1.1, beta = 11., bnb = True, measure_time = False):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers

    m = g.Model("miqp")
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    if (measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
        # m.params.mipfocus = 2
        # m.params.nodelimit = 1000000
        # m.params.branchdir = -1
    if (bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)


    degree = np.zeros((H_graph.nodes.shape[0],1))
    ones_edges = np.ones_like(H_graph.edges)
    np.add.at(degree, H_graph.receivers, ones_edges)
    max_degree = int(np.max(degree)) + 1 ### TODO think about why this +1 is neccesary here in case of a fully conencted graph
    num_additional_neurons = int(np.log2(max_degree)) + 1

    B = 1
    A = (max_degree + 2)*B
    C = B

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)
    var_dict_y = m.addVars(max_degree - 1, vtype=g.GRB.BINARY)

    sum_nodes = g.quicksum( var_dict[int(n)]  for n in range(N))
    sum_n_times_y = g.quicksum( 2**n*var_dict_y[int(n)]  for n in range(num_additional_neurons))
    HA2 = A*(sum_n_times_y- sum_nodes)**2
    HA = HA2

    HB11 = g.quicksum( 2**n*var_dict_y[int(n)]  for n in range(num_additional_neurons))
    HB1 = 0.5*B*HB11 *(HB11 - 1)
    HB2 = B*g.quicksum( -1/2*var_dict[int(s)] * var_dict[int(r)] for s, r, in zip(senders, receivers))
    HB = HB1 + HB2
    HC = -C*g.quicksum( var_dict[int(n)]  for n in range(N))

    m.setObjective(HA + HB + HC, g.GRB.MINIMIZE)
    m.optimize()

    print("MaxCliqueSize",m.ObjVal)
    y_sol = np.array([var_dict_y[key].x for key in var_dict_y])
    return m, m.ObjVal, np.array([var_dict[key].x for key in var_dict]),  m.Runtime

def solveMVC(H_graph, time_limit=float("inf"), solution_limit=None, verbose=0, A = 1., B = 1.1, bnb = True, measure_time = False):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    B = B
    A = A

    m = g.Model("mip1")
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Threads", int(0.6 * multiprocessing.cpu_count()))
    if (measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)

    if (bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)

    obj1 = B*g.quicksum( B/2*(1 - var_dict[int(s)]) * (1 - var_dict[int(r)]) for s, r, in zip(senders, receivers))
    obj2 = g.quicksum( A* var_dict[int(n)]  for n in range(N))

    m.setObjective(obj1 + obj2, g.GRB.MINIMIZE)
    m.optimize()

    bins = np.array([var_dict[key].x for key in var_dict])
    bins = np.expand_dims(bins, axis = -1)
    print("Energy check")
    if(False):
        num_edges = H_graph.edges.shape[0]

        ### TODO H_graph.nodes.shape is weirt pls fix it
        degree = np.zeros((H_graph.nodes.shape[0],1))
        ones_edges = np.ones_like(H_graph.edges)
        np.add.at(degree, H_graph.receivers, ones_edges)

        two_body_corr = np.sum(bins[senders]*bins[receivers])

        print("here1")
        print(np.sum(degree))
        print(H_graph.senders.shape)

        Ha = A*np.sum(bins)
        Hb = B*(num_edges/2 - np.sum(bins*degree)+ two_body_corr/2)


        print("MinVertexCoverSize",m.ObjVal, Ha + Hb)
    else:
        print("MinVertexCoverSize",m.ObjVal)
    return m, m.ObjVal, np.squeeze(bins, axis = -1),  m.Runtime

def solveMVC_continuous(H_graph, time_limit=float("inf"), solution_limit=None, verbose=0, A = 1., B = 1.1, bnb = True, measure_time = False):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    B = B
    A = A

    m = g.Model("mip1")
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Threads", int(0.6 * multiprocessing.cpu_count()))
    m.setParam("NonConvex", 2)
    if (measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)

    if (bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    var_dict = m.addVars(N, vtype=g.GRB.CONTINUOUS, ub = 1., lb = 0.)

    obj1 = B*g.quicksum( B/2*(1 - var_dict[int(s)]) * (1 - var_dict[int(r)]) for s, r, in zip(senders, receivers))
    obj2 = g.quicksum( A* var_dict[int(n)]  for n in range(N))

    m.setObjective(obj1 + obj2, g.GRB.MINIMIZE)
    m.optimize()

    bins = np.array([var_dict[key].x for key in var_dict])
    bins = np.expand_dims(bins, axis = -1)
    print("Energy check")
    if(False):
        num_edges = H_graph.edges.shape[0]

        ### TODO H_graph.nodes.shape is weirt pls fix it
        degree = np.zeros((H_graph.nodes.shape[0],1))
        ones_edges = np.ones_like(H_graph.edges)
        np.add.at(degree, H_graph.receivers, ones_edges)

        two_body_corr = np.sum(bins[senders]*bins[receivers])

        print("here1")
        print(np.sum(degree))
        print(H_graph.senders.shape)

        Ha = A*np.sum(bins)
        Hb = B*(num_edges/2 - np.sum(bins*degree)+ two_body_corr/2)


        print("MinVertexCoverSize",m.ObjVal, Ha + Hb)
    else:
        print("MinVertexCoverSize",m.ObjVal)
    return m, m.ObjVal, np.squeeze(bins, axis = -1),  m.Runtime

def solveMIS(H_graph, time_limit=float("inf"), model_name = "miqp1", solution_limit=None, verbose=0, A = 1., B = 3., bnb = True, measure_time = False):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    B = B
    A = A

    m = g.Model(model_name)
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)
    if (measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
        # m.params.mipfocus = 2
        # m.params.nodelimit = 1000000
        # m.params.branchdir = -1
    else:
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(0.75*multiprocessing.cpu_count()))
    if (bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)

    obj1 = B*g.quicksum( B*var_dict[int(s)] * var_dict[int(r)] for s, r, in zip(senders, receivers))
    obj2 = g.quicksum( - A* var_dict[int(n)] * var_dict[int(n)]  for n in range(N))

    m.setObjective(obj1 + obj2, g.GRB.MINIMIZE)
    m.optimize()

    print("MIS solution", m.ObjVal)
    return m, m.ObjVal, np.array([var_dict[key].x for key in var_dict]),  m.Runtime

def solveMaxCut(H_graph, time_limit=float("inf"), solution_limit=None, num_CPUs = None, verbose=0, measure_time = False, bnb = False, model_name = "mip1", thread_fraction = 0.75):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    edges = H_graph.edges

    m = g.Model(model_name)
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)

    if(measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
    elif(num_CPUs == None):
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(thread_fraction*multiprocessing.cpu_count()))
    else:
        m.setParam("Threads", int(num_CPUs))

    if(bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)

    obj1 = g.quicksum( 0.5*(2*var_dict[int(s)]-1) *weight* (2*var_dict[int(r)]-1) for s, r, weight in zip(senders, receivers, edges))
    #obj2 = g.quicksum(  (2*var_dict[int(n)]-1) * (2*var_dict[int(n)]-1)/4  for n in range(N))

    m.setObjective(obj1, g.GRB.MINIMIZE)
    m.optimize()
    if(measure_time):
        print(m.printStats())

    solutions = np.array([var_dict[key].x for key in var_dict])
    MaxCut_value = sum([ (1-(2*solutions[int(s)]-1) *weight* (2*solutions[int(r)]-1))/4 for s, r, weight in zip(senders, receivers, edges)])

    return m, m.ObjVal, m.objBound, np.array([var_dict[key].x for key in var_dict]), m.Runtime, MaxCut_value

def solveSpinGlass(H_graph, time_limit=float("inf"), solution_limit=None, num_CPUs = None, verbose=0, measure_time = False, bnb = False, model_name = "mip1", thread_fraction = 0.75):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    edges = H_graph.edges

    m = g.Model(model_name)
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)

    if(measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
    elif(num_CPUs == None):
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(thread_fraction*multiprocessing.cpu_count()))
    else:
        m.setParam("Threads", int(num_CPUs))

    if(bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    var_dict = m.addVars(N, vtype=g.GRB.BINARY)

    obj1 = g.quicksum( -0.5*(2*var_dict[int(s)]-1) *weight* (2*var_dict[int(r)]-1) for s, r, weight in zip(senders, receivers, edges))
    #obj2 = g.quicksum(  (2*var_dict[int(n)]-1) * (2*var_dict[int(n)]-1)/4  for n in range(N))

    m.setObjective(obj1, g.GRB.MINIMIZE)
    m.optimize()

    return m, m.ObjVal, m.objBound, np.array([var_dict[key].x for key in var_dict]), m.Runtime


def solveMaxCut_as_IP(H_graph, time_limit=float("inf"), solution_limit=None, verbose=0, measure_time = False, bnb = True, model_name = "mip1"):
    N = int(H_graph.n_node[0])
    senders = H_graph.senders
    receivers = H_graph.receivers
    edges = H_graph.edges

    m = g.Model(model_name)
    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", time_limit)

    multiprocessing.cpu_count()
    if(measure_time):
        m.setParam("Threads", 1)
        m.setParam("OutputFlag", 1)
    else:
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(0.75*multiprocessing.cpu_count()))

    if(bnb):
        m.setParam("Heuristics", 0)
        m.setParam("Cuts", 0)
        m.setParam("RINS", 0)
        m.setParam("Presolve", 0)
        m.setParam("Aggregate", 0)
        m.setParam("Symmetry", 0)
        m.setParam("Disconnected", 0)
    if not isinstance(solution_limit, type(None)):
        m.setParam("SolutionLimit", solution_limit)

    # var_dict = {}
    # for i in range(N):
    #     var_dict[i] = m.addVar(vtype=g.GRB.BINARY, name=f'{i}')

    node_var_dict = m.addVars(N, vtype=g.GRB.BINARY)
    edge_var_dict = m.addVars(senders.shape[0], vtype=g.GRB.BINARY)

    obj1 = g.quicksum( -edge_var_dict[idx]/2 for idx, (s, r, weight) in enumerate(zip(senders, receivers, edges)))
    #obj2 = g.quicksum(  (2*var_dict[int(n)]-1) * (2*var_dict[int(n)]-1)/4  for n in range(N))
    for idx, (s, r, weight) in enumerate(zip(senders, receivers, edges)):
        m.addConstr(edge_var_dict[idx]  <=  node_var_dict[s] + node_var_dict[r])
        m.addConstr(edge_var_dict[idx] <= 2 - (node_var_dict[s] + node_var_dict[r]))

    m.setObjective(obj1, g.GRB.MINIMIZE)
    m.optimize()
    if(measure_time):
        print(m.printStats())

    MCEnergy = sum([ ((2*node_var_dict[int(s)].x-1) *weight* (2*node_var_dict[int(r)].x-1))/2 for s, r, weight in zip(senders, receivers, edges)])
    MCValue = 0.5*sum([ (1 -(2*node_var_dict[int(s)].x-1) *weight* (2*node_var_dict[int(r)].x-1))/2 for s, r, weight in zip(senders, receivers, edges)])
    return m, m.ObjVal, m.objBound, MCEnergy, np.array([node_var_dict[key].x for key in node_var_dict]), m.Runtime
