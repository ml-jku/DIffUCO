import pickle
import os

def save_indexed_dict(path: str, mode: str , dataset_name: str, i: int, EnergyFunction: str, seed: int, indexed_solution_dict):

    newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}/{seed}/{EnergyFunction}/indexed"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}/{seed}/{EnergyFunction}/indexed/idx_{i}_solutions.pickle"
    pickle.dump(indexed_solution_dict, open(save_path, "wb"))

def load_indexed_dict(path: str, mode: str , dataset_name: str, i: int, EnergyFunction: str, seed: int):

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}/{seed}/{EnergyFunction}/indexed/idx_{i}_solutions.pickle"
    loaded_dict = pickle.load( open(save_path, "rb"))
    return loaded_dict
