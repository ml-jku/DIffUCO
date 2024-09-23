import pickle
import numpy as np
def compute_average_metrics(seed_path, N = 2000):

    n_edges = {"train_graph": [], "Energy_graph": []}
    n_nodes = {"train_graph": [], "Energy_graph": []}
    for i in range(N):
        file_idx = i
        file_path = f"idx_{i}_solutions.pickle"

        overall_path = seed_path + file_path

        with open(overall_path, "rb") as f:
            res = pickle.load(f)

        n_edges["train_graph"].append(res['H_graphs'].edges.shape[0])
        n_edges["Energy_graph"].append(res['compl_H_graphs'].edges.shape[0])
        n_nodes["Energy_graph"].append(res['compl_H_graphs'].nodes.shape[0])
        n_nodes["train_graph"].append(res['H_graphs'].nodes.shape[0])

    print("average metrics")
    print("mean nodes train ", np.mean(n_nodes["train_graph"]) , np.std(n_nodes["train_graph"]))
    print("mean edges train", np.mean(n_edges["train_graph"]), np.std(n_edges["train_graph"]) )
    print("mean nodes energy ", np.mean(n_nodes["Energy_graph"]),np.std(n_nodes["Energy_graph"]) )
    print("mean edges energy", np.mean(n_edges["Energy_graph"]),np.std(n_edges["Energy_graph"]) )



if(__name__ == "__main__"):


    seed = 123
    path_123 = f"/system/user/publicwork/sanokows/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/RB_iid_200/train/{seed}/MVC/indexed/"
    seed_2 = 124
    path_124 = f"/system/user/publicwork/sanokows/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/RB_iid_200/train/{seed_2}/MVC/indexed/"
    seed_3 = 125
    path_125 = f"/system/user/publicwork/sanokows/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/RB_iid_200/train/{seed_3}/MVC/indexed/"

    print("seed 123")
    compute_average_metrics(path_123)
    print("seed 124")
    compute_average_metrics(path_124)
    print("seed 125")
    compute_average_metrics(path_125)

