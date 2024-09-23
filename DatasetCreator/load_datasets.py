import sys
sys.path.append("..")

from DatasetCreator.loadGraphDatasets import get_dataset_generator
import numpy as np
import argparse

import time
import numpy as np

RB_datasets = ["RB_iid_200", "RB_iid_100", "RB_iid_small", "RB_iid_large", "RB_iid_giant", "RB_iid_huge", "RB_iid_dummy"]
BA_datasets = ["BA_small", "BA_large", "BA_huge", "BA_giant", "BA_dummy"]
TSP_datasets = ['TSP_random_100', "TSP_random_20"]
Gset = ["Gset"]
IsingModel = ["NxNLattice_4x4", "NxNLattice_8x8", "NxNLattice_16x16", "NxNLattice_24x24", "NxNLattice_32x32"]
dataset_choices =  RB_datasets + BA_datasets + TSP_datasets + Gset + IsingModel
parser = argparse.ArgumentParser()

parser.add_argument('--licence_path', default="/system/user/sanokows/", type = str, help='licence base path')
parser.add_argument('--seed', default=[123, 124, 125], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--parent', default=False, type = bool, help='use parent directory or not')
parser.add_argument('--save', default=False, type = bool, help='save the entire dataset in a pickle file or not')
parser.add_argument('--gurobi_solve', default=True, type = bool, help='whether to solve instances with gurobi or not')
parser.add_argument('--datasets', default=['RB_iid_large'], choices = dataset_choices, help='Define the dataset', nargs="+")
parser.add_argument('--diff_ps', default=False, type = bool, help='')
parser.add_argument('--problems', default=['MIS'], choices = ["MIS", "MVC", "MaxCl", "MaxCut", "MDS", "TSP", "IsingModel"], help='Define the CO problem', nargs="+")
parser.add_argument('--mode', default= "test", type = str, help='Define dataset split')
parser.add_argument('--time_limits', default=["inf"], type = str, help='Gurobi Time Limit for each [mode]', nargs = "+")
#parser.add_argument('--n_graphs', default=[100, 10, 10], type = int, help='Number of graphs for each [mode]', nargs = "+")
args = parser.parse_args()


def load_dataset(config: dict, modes: list, time_limits: list):
	"""
	Create a dataset with the specified configuration

	:param config: config dictionary specifying the dataset that should be generated
	:param modes: ["train", "val", "test"] modes for which the dataset should be generated
	:param sizes: [int, int, int] number of graphs for each mode
	:param time_limits: [float, float, float] time limit for each mode
	"""

	config["mode"] = modes
	config["thread_fraction"] = None
	config["time_limit"] = None
	dataset_generator = get_dataset_generator(config)
	return dataset_generator.load_solutions()


def seconds_to_hms(seconds):
	# Calculate hours, minutes, and remaining seconds
	hours = seconds // 3600
	minutes = (seconds % 3600) // 60
	seconds = seconds % 60

	# Return the formatted string as H:M:S
	return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# python prepare_datasets.py --licence_path /system/user/berghamm --datasets NxNLattice_16x16 --problems IsingModel --seed 123 --save True --gurobi_solve False --modes test train val
if __name__ == "__main__":

	for dataset in args.datasets:
		for problem in args.problems:
			solutions_dict = {"runtimes": [], "Energies": []}
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
				sol_per_seed_dict = load_dataset(base_config, args.mode, args.time_limits)

				for key in solutions_dict.keys():
					solutions_dict[key].append(sol_per_seed_dict[key])

			for key in solutions_dict.keys():
				if(key == "runtimes"):
					mean_time =  time.strftime('%H:%M:%S', time.gmtime(np.mean(solutions_dict[key])))
					mean_time =  seconds_to_hms(np.mean(solutions_dict[key]))
					std_time = time.strftime('%H:%M:%S', time.gmtime(np.mean(np.std(solutions_dict[key]) / np.sqrt(len(args.seed)))))
					print(solutions_dict[key])
					print(key, "$", mean_time, "\pm",std_time, "$")
				else:
					print(solutions_dict[key])
					print(key, "$", np.round(np.mean(solutions_dict[key]), 2), "\pm", np.round(np.std(solutions_dict[key])/np.sqrt(len(args.seed)),3) , "$")