import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import imageio
import igraph as ig
from DatasetCreator.Gurobi.GurobiSolver import solveMaxCut, solveMIS_as_MIP
from DatasetCreator.jraph_utils.utils import from_igraph_to_jgraph
from DatasetCreator.loadGraphDatasets.RB_graphs import generate_xu_instances
def beta_schedule(n_diff_steps):

    beta_list = []
    for i in range(n_diff_steps):
        beta = 1/(2+i)
        beta_list.append(beta)
    return np.flip(beta_list)


def flip_states(states, beta):
    p = np.random.uniform(size=len(states))
    spins = 2*states -1

    flipped_spins = np.where(p < beta, -1*spins, spins)
    flipped_bins = (flipped_spins+1)/2

    print("num of flipped states", np.abs(flipped_bins - states).sum())
    return flipped_bins

def plot(er_graph, images_list):
    node_features = nx.get_node_attributes(er_graph, 'feature')

    # Partition nodes based on features
    nodes_0 = [node for node, feature in node_features.items() if feature == 0]
    nodes_1 = [node for node, feature in node_features.items() if feature == 1]


    edges_between_0_and_1 = 0
    for node_0 in nodes_0:
        for node_1 in nodes_1:
            if er_graph.has_edge(node_0, node_1):
                edges_between_0_and_1 += 1

    print("Number of edges between nodes with feature 0 and feature 1:", edges_between_0_and_1)

    # Position nodes
    pos = {}
    f = 0.5
    for i, node in enumerate(nodes_0):
        r = f * (np.random.uniform() - 0.5)
        r_2 = f * (np.random.uniform() - 0.5)
        pos[node] = (1 + r, i + r_2)
    for i, node in enumerate(nodes_1):
        r = f * (np.random.uniform() - 0.5)
        r_2 = f * (np.random.uniform() - 0.5)
        pos[node] = (2 + r, i + r_2)

    # Draw the graph
    plt.figure(figsize=(10, 6))

    # Draw nodes
    nx.draw_networkx_nodes(er_graph, pos, nodelist=nodes_0, node_color='skyblue', node_size=800)
    nx.draw_networkx_nodes(er_graph, pos, nodelist=nodes_1, node_color='salmon', node_size=800)

    # Draw edges
    nx.draw_networkx_edges(er_graph, pos, edge_color='gray')

    # Draw labels
    nx.draw_networkx_labels(er_graph, pos, font_color='black', font_size=12)

    plt.title(f"Erdős-Rényi Graph with Random Node Features: num cuts {edges_between_0_and_1}")
    plt.xticks([])  # Hide x-axis
    plt.yticks([])  # Hide y-axis
    plt.savefig('temp_plot.png')
    plt.show()
    plt.close()
    images_list.append(imageio.imread('temp_plot.png'))
    return images_list

def plot_normal(er_graph, images_list):
    node_features = nx.get_node_attributes(er_graph, 'feature')

    # Partition nodes based on features
    nodes_0 = [node for node, feature in node_features.items() if feature == 0]
    nodes_1 = [node for node, feature in node_features.items() if feature == 1]

    def get_node_color(graph, node):
        if graph.nodes[node]["feature"] == 0:
            return "skyblue"
        else:
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if graph.nodes[neighbor]["feature"] == 1:
                    return "limegreen"
            return "limegreen"

    def get_edge_color(graph, edge):
        u, v = edge
        if graph.nodes[u]["feature"] == 1 and graph.nodes[v]["feature"] == 1:
            return "red"
        else:
            return "black"

    # Set edge colors based on neighboring node features
    edge_colors = [get_edge_color(er_graph, edge) for edge in er_graph.edges()]

    def compute_violations(graph, node):
        if graph.nodes[node]["feature"] == 0:
            return "skyblue"
        else:
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if graph.nodes[neighbor]["feature"] == 1:
                    return "limegreen"
            return "green"

    # Set node colors based on features and neighbors
    node_colors = [get_node_color(er_graph, node) for node in er_graph.nodes()]
    num_violations = np.sum([1*(get_edge_color(er_graph, edge) == "red") for edge in er_graph.edges()])

    MIS_size = len(nodes_1)
    node_labels = {node: int(er_graph.nodes[node]["feature"]) for node in er_graph.nodes()}
    # Draw the graph
    node_features = nx.get_node_attributes(er_graph, 'feature')
    #node_colors = ['skyblue' if feature == 0 else 'salmon' for feature in node_features.values()]

    plt.figure(figsize=(6, 6))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.title(f"Set size: {MIS_size} \n #violations: {num_violations}", fontsize=50)
    pos = nx.spring_layout(er_graph, seed = 42)
    nx.draw_networkx_edges(er_graph, pos, width=3, edge_color=edge_colors)
    nx.draw(er_graph, pos = pos, labels=node_labels, node_color=node_colors, edge_color=edge_colors, node_size=1600, edgecolors="black")
    plt.tight_layout()
    plt.xticks([])  # Hide x-axis
    plt.yticks([])  # Hide y-axis
    plt.savefig('temp_plot.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.title(f"Set size: {MIS_size} \n #violations: {num_violations}", fontsize=50)
    pos = nx.spring_layout(er_graph, seed = 42)
    nx.draw_networkx_edges(er_graph, pos, width=3, edge_color=["black" for edge in er_graph.edges()])
    nx.draw(er_graph, pos = pos, with_labels=False, node_color=["skyblue" for node in er_graph.nodes()], node_size=1600, edgecolors="black")
    plt.tight_layout()
    plt.xticks([])  # Hide x-axis
    plt.yticks([])  # Hide y-axis
    plt.savefig('temp_plot.png')
    plt.show()
    plt.close()

    images_list.append(imageio.imread('temp_plot.png'))
    return images_list

if(__name__ == "__main__"):

    b = 100
    a = 0.01
    Taus = [20, 50, 10, 150, 500]
    plt.figure()

    for Tau in Taus:
        betas = []
        for i in range(Tau):
            t = Tau - i
            beta = 0.5 * (1 - np.exp(Tau * a * (b ** (t / Tau) - b ** ((t + 1) / Tau))))
            beta = max([beta, 0.01])
            betas.append(beta)

        plt.plot(np.arange(0, Tau)/Tau, betas, "-x", label = f"Tau = {Tau}")
    plt.legend()
    plt.show()

    raise ValueError("")



    # Define grid dimensions
    rows = 5
    cols = 5

    # Create a 2D grid graph
    G = nx.grid_2d_graph(rows, cols, periodic=True)

    # Create a dictionary of positions for each node
    pos = {(x, y): (y, -x) for x, y in G.nodes()}

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    plt.title("2D Grid Graph")
    plt.axis('equal')
    plt.show()
    print([node for node in G.nodes()])

    num_vertices = G.number_of_nodes()

    node_idx = {}
    for i, node in enumerate(G.nodes):
        node_idx[node] = i

    edge_idx = []
    for i, edge in enumerate(G.edges):
        sender, receiver = edge
        edge_idx.append([node_idx[sender], node_idx[receiver]])

    pass
    # for p in np.linspace(0.7, 1., 10):
    #     n = np.random.randint(5, 10)
    #     k = np.random.randint(5, 10)
    #
    #     edges = generate_xu_instances.get_random_instance(n, k, p)
    #
    #     import networkx as nx
    #     from matplotlib import pyplot as plt
    #     G = nx.Graph()
    #     G.add_edges_from(edges)
    #     nx.draw(G, node_size=30)
    #     plt.show()
    #
    # raise ValueError()

    # Generate an Erdős-Rényi graph
    # n = 20  # Number of nodes
    # p = 0.2  # Probability of edge creation
    # n_diffusion_steps = 10
    # er_graph = nx.erdos_renyi_graph(n, p)
    # er_graph = nx.barabasi_albert_graph(n, 4)
    #
    # igraph = ig.Graph()
    # igraph.add_vertices(er_graph.nodes())
    # igraph.add_edges(er_graph.edges())
    np.random.seed(42)
    n = 12#np.random.randint(10, 20)
    k = 10
    p = 0.5
    n = 2#np.random.randint(10, 20)
    k = 6
    p = 0.75
    n_diffusion_steps = 7
    beta_list = beta_schedule(n_diffusion_steps)

    edges = generate_xu_instances.get_random_instance(n, k, p)
    g = ig.Graph([(edge[0], edge[1]) for edge in edges])
    isolated_nodes = [v.index for v in g.vs if v.degree() == 0]
    g.delete_vertices(isolated_nodes)
    num_nodes = g.vcount()
    print("num nodes:", num_nodes)
    n = 15  # Number of nodes
    p = 0.2  # Probability of edge creation
    er_graph = nx.erdos_renyi_graph(n, p)
    er_graph = nx.barabasi_albert_graph(n, 4)
    g = ig.Graph([(edge[0], edge[1]) for edge in er_graph.edges])
    # nxg = nx.Graph()
    # nxg.add_edges_from(edges)
    # er_graph = nxg


    H_graph = from_igraph_to_jgraph(g)
    res = solveMIS_as_MIP(H_graph)
    gt_bins = res[2]

    for idx, node in enumerate(er_graph.nodes):

        er_graph.nodes[node]['feature'] = gt_bins[idx]

    er_graph_list = []
    for i in range(n_diffusion_steps):
        numpy_node_features = np.array([er_graph.nodes[node]['feature'] for node in er_graph.nodes])
        flipped_node_features = flip_states(numpy_node_features, beta_list[i])

        er_copy = copy.deepcopy(er_graph)
        for idx, node in enumerate(er_graph.nodes):
            er_graph.nodes[node]['feature'] = flipped_node_features[idx]
        er_graph_list.append(er_copy)

    images_list = []
    for er_graph in er_graph_list:
        images_list = plot_normal(er_graph, images_list)

    images_list.reverse()
    filename = "test_gif.gif"
    imageio.mimsave(filename, images_list, duration=800)
    print(f"GIF saved as {filename}", len(images_list))