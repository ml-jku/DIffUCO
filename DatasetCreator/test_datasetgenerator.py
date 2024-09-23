import sys
sys.path.append("..")

from DatasetCreator.loadGraphDatasets import get_dataset_generator

import argparse

RB_datasets = ["RB_iid_200", "RB_iid_100", "RB_iid_small", "RB_iid_large", "RB_iid_giant", "RB_iid_huge", "RB_iid_100_dummy"]
BA_datasets = ["BA_small", "BA_large", "BA_huge", "BA_giant"]
TSP_datasets = ['2D_random_100', "2D_random_20"]
Gset = ["Gset"]
dataset_choices =  RB_datasets + BA_datasets + TSP_datasets + Gset
parser = argparse.ArgumentParser()

parser.add_argument('--licence_path', default="/system/user/sanokows/", type = str, help='licence base path')
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--parent', default=False, type = bool, help='use parent directory or not')
parser.add_argument('--save', default=True, type = bool, help='save the entire dataset in a pickle file or not')
parser.add_argument('--gurobi_solve', default=True, type = bool, help='whether to solve instances with gurobi or not')
parser.add_argument('--datasets', default=['Gset'], choices = dataset_choices, help='Define the dataset', nargs="+")
parser.add_argument('--diff_ps', default=False, type = bool, help='')
parser.add_argument('--problems', default=['MaxCut'], choices = ["MIS", "MVC", "MaxCl", "MaxCut", "MDS", "TSP"], help='Define the CO problem', nargs="+")
parser.add_argument('--modes', default=[ "test", "train", "val"], type = str, help='Define dataset split', nargs = "+")
parser.add_argument('--time_limits', default=["inf", "0.5", "0.5"], type = str, help='Gurobi Time Limit for each [mode]', nargs = "+")
#parser.add_argument('--n_graphs', default=[100, 10, 10], type = int, help='Number of graphs for each [mode]', nargs = "+")
args = parser.parse_args()


def create_dataset(config: dict, modes: list, time_limits: list):
	"""
	Create a dataset with the specified configuration

	:param config: config dictionary specifying the dataset that should be generated
	:param modes: ["train", "val", "test"] modes for which the dataset should be generated
	:param sizes: [int, int, int] number of graphs for each mode
	:param time_limits: [float, float, float] time limit for each mode
	"""
	if len(modes) != len(time_limits):
		raise ValueError("Length of modes, sizes and time_limits should be the same")

	for mode,  time_limit in zip(modes, time_limits):
		config["mode"] = mode
		#config["n_graphs"] = size
		config["time_limit"] = float(time_limit)
		dataset_generator = get_dataset_generator(config)
		dataset_generator.generate_dataset()


if __name__ == "__main__":
	for dataset in args.datasets:
		for problem in args.problems:
			for seed in args.seed:
				base_config = {
					"licence_base_path": args.licence_path,
					"seed": seed,
					"parent": args.parent,
					"save": args.save,
					"gurobi_solve": args.gurobi_solve,
					"diff_ps": args.diff_ps,
					"dataset_name": dataset,
					"problem": problem,
					"time_limit": None,
					"n_graphs": None,
				}
				create_dataset(base_config, args.mode, args.time_limits)