
import pickle
import os

def read_and_get_mean_energy(wandb_run_id, eval_steps = 3, n_states = 8, mode = "test"):

    current_file_path = os.path.abspath(__file__)

    # Get the parent directory of the current file
    parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    path_to_models = parent_folder + "/Checkpoints"
    path_folder = f"{path_to_models}/{wandb_run_id}/"

    if not os.path.exists(path_folder):
        print(f"Folder not found: {path_folder}")
        raise ValueError("Folder not found")

    file_name = f"{wandb_run_id}_test_dict_eval_step_factor_{eval_steps}_{n_states}.pickle"

    # Read the pickle file
    with open(os.path.join(path_folder, file_name), 'rb') as f:
        test_dict = pickle.load(f)

    if('test/MaxCut_Value_CE' in test_dict.keys()):
        result_dict = {"mean_energy_CE": test_dict["test/MaxCut_Value_CE"],
                       "mean_energy": test_dict["test/MaxCut_Value"],
                       "overall_time": test_dict["test/overall_time"],
                       "forward_pass_time": test_dict["test/forward_pass_time"]
                       }
    else:
        result_dict = {"mean_energy_CE": test_dict["test/mean_energy_CE"],
                       "mean_energy": test_dict["test/mean_energy"],
                       "overall_time": test_dict["test/overall_time"],
                       "CE_time": test_dict["test/CE_time"],
                       "forward_pass_time": test_dict["test/forward_pass_time"]
                       }

    return result_dict

def read_and_get_best_energy(wandb_run_id, eval_steps = 3, n_states = 150, mode = "test"):
    current_file_path = os.path.abspath(__file__)

    # Get the parent directory of the current file
    parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    path_to_models = parent_folder + "/Checkpoints"
    path_folder = f"{path_to_models}/{wandb_run_id}/"

    if not os.path.exists(path_folder):
        print(f"Folder not found: {path_folder}")
        raise ValueError("Folder not found")

    file_name = f"{wandb_run_id}_test_dict_eval_step_factor_{eval_steps}_{n_states}.pickle"

    # Read the pickle file
    with open(os.path.join(path_folder, file_name), 'rb') as f:
        test_dict = pickle.load(f)

    if('test/best_MaxCut_Value_CE' in test_dict.keys()):
        result_dict = {"best_energy_CE": test_dict["test/best_MaxCut_Value_CE"],
                       "best_energy": test_dict["test/best_MaxCut_Value_CE"],
                       }
    else:
        result_dict = {"best_energy_CE": test_dict["test/mean_best_energy_CE"],
                       "best_energy": test_dict["test/mean_best_energy"],
                       "mean_gt_energy": test_dict['test/mean_gt_energy'],
                       }

    return result_dict

diff_steps = [4,8,12,16]
MIS_small_fKL = ["azx4ctuf", "wno5zlcg", "lf9b0glc", "5ck9ydrq"]
MIS_small_rKL = ["sqgmhosx", "k27lzwki", "lm6nwmd9", "gliujcf1"]
MIS_small_PPO = ["1e119d1y", "29oajlej", "shhasbii", "tsdgkzq8"]
MIS_small_Scaling_dict = {
    "PPO": MIS_small_PPO,
    "rKL": MIS_small_rKL,
    "fKL": MIS_small_fKL,
}
fKL_results = {"mean": [], "best_mean": [], "rel_error": [], "best_rel_error": []}
rKL_results = {"mean": [], "best_mean": [], "rel_error": [], "best_rel_error": []}
PPO_results = {"mean": [], "best_mean": [], "rel_error": [], "best_rel_error": []}


for key, run_ids in MIS_small_Scaling_dict.items():
    for run_id in run_ids:
        results_dict = read_and_get_mean_energy(run_id, eval_steps = 1, n_states = 30)
        mean_energy = results_dict["mean_energy"]
        
        results_dict = read_and_get_best_energy(run_id, eval_steps = 1, n_states = 60)
        best_mean_energy = results_dict["best_energy"]
        gt_energy = results_dict["mean_gt_energy"]
        if key == "fKL":
            fKL_results["mean"].append(mean_energy)
            fKL_results["best_mean"].append(best_mean_energy)
            fKL_results["rel_error"].append(abs(mean_energy - gt_energy)/abs(gt_energy))
            fKL_results["best_rel_error"].append(abs(best_mean_energy - gt_energy)/abs(gt_energy))
        elif key == "rKL":
            rKL_results["mean"].append(mean_energy)
            rKL_results["best_mean"].append(best_mean_energy)
            rKL_results["rel_error"].append(abs(mean_energy - gt_energy)/abs(gt_energy))
            rKL_results["best_rel_error"].append(abs(best_mean_energy - gt_energy)/abs(gt_energy))
        elif key == "PPO":
            PPO_results["mean"].append(mean_energy)
            PPO_results["best_mean"].append(best_mean_energy)
            PPO_results["rel_error"].append(abs(mean_energy - gt_energy)/abs(gt_energy))
            PPO_results["best_rel_error"].append(abs(best_mean_energy - gt_energy)/abs(gt_energy))  

import matplotlib.pyplot as plt

            # Calculate the mean of the results
            # Calculate the mean of the results
fKL_marker_size = 6
rKL_marker_sizes = [int(step*6/4) for step in diff_steps]
PPO_marker_size = 6

# Plotting
plt.figure()
plt.plot(diff_steps, fKL_results["mean"], label='fKL', marker='o', markersize=fKL_marker_size)
plt.plot(diff_steps, rKL_results["mean"], label='rKL', color = "red")
for i, txt in enumerate(diff_steps):
    plt.plot(diff_steps[i], rKL_results["mean"][i], marker = "<", markersize = rKL_marker_sizes[i], color = "red")
plt.plot(diff_steps, PPO_results["mean"], label='PPO', marker='^', markersize=PPO_marker_size)
plt.axhline(y=gt_energy, color='r', linestyle='-', label='Ground Truth')
plt.xlabel('Diffusion Steps')
plt.ylabel('Mean Energy')
plt.title('Mean Energy vs Diffusion Steps')
plt.grid()
plt.legend()

# Save the plot
output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, 'mean_energy_vs_diff_steps.png')
plt.savefig(output_path)
plt.close()

plt.figure()
plt.plot(diff_steps, fKL_results["best_mean"], label='fKL', marker='o', markersize=fKL_marker_size)
plt.plot(diff_steps, rKL_results["best_mean"], label='rKL', color = "red")

for i, txt in enumerate(diff_steps):
    plt.plot(diff_steps[i], rKL_results["best_mean"][i], marker = "<", markersize = rKL_marker_sizes[i], color = "red")
plt.plot(diff_steps, PPO_results["best_mean"], label='PPO', marker='^', markersize=PPO_marker_size)
plt.xlabel('Diffusion Steps')
plt.ylabel('Best Energy')

plt.axhline(y=gt_energy, color='r', linestyle='-', label='Ground Truth')
plt.grid()
plt.title('Best Energy vs Diffusion Steps')
plt.legend()

# Save the plot
output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, 'best_energy_vs_diff_steps.png')
plt.savefig(output_path)
plt.close()

from matplotlib.ticker import FuncFormatter

# Function to format y-axis ticks
def ytick_formatter(y, _):
    print("here", y, _, f'{y:.3f}')
    return f'{y:.3f}'

plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
# Generate the plot
plt.figure()
#plt.tick_params(axis='both', which='minor', labelsize=20)
plt.plot(diff_steps, fKL_results["rel_error"], label='SDDS: fKL w/ MC', marker='o', markersize=fKL_marker_size)
plt.plot(diff_steps, rKL_results["rel_error"], label='DiffUCO', color="red", marker="<", markersize=fKL_marker_size)
for i, txt in enumerate(diff_steps):
    plt.plot(diff_steps[i], rKL_results["rel_error"][i], marker="<", markersize=rKL_marker_sizes[i], color="red")
plt.plot(diff_steps, PPO_results["rel_error"], label='SDDS: rKL w/ RL', marker='^', markersize=PPO_marker_size, color="green")

# Label axes
plt.xlabel('Diffusion Steps', fontsize = 14)
plt.ylabel(r'$\epsilon_{\mathrm{rel}}$', fontsize=24)
plt.ylim(top = 0.115)
# Log-scale for y-axis and custom formatting
plt.yscale('log')
#plt.gca().yaxis.set_major_formatter(FuncFormatter(ytick_formatter))

# Ensure gridlines are shown on both axes
plt.grid(visible=True, which='both', linestyle='-', alpha=0.7, linewidth=2)

# Add legend and tighten layout
plt.legend(loc = "upper right", fontsize = 16)
plt.tight_layout()

# Save the plot
output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, 'rel_error_vs_diff_steps.png')
plt.savefig(output_path, dpi=1200)
plt.close()

plt.figure()
plt.plot(diff_steps, fKL_results["best_rel_error"], label='SDDS: fKL w/ MC', marker='o', markersize=fKL_marker_size)
plt.plot(diff_steps, rKL_results["best_rel_error"], label='DiffUCO', color="red", marker="<", markersize=fKL_marker_size)
for i, txt in enumerate(diff_steps):
    plt.plot(diff_steps[i], rKL_results["best_rel_error"][i], marker="<", markersize=rKL_marker_sizes[i], color="red")
plt.plot(diff_steps, PPO_results["best_rel_error"], label='SDDS: rKL w/ RL', marker='^', markersize=PPO_marker_size, color="green")

# Label axes
plt.xlabel('Diffusion Steps', fontsize = 14)
plt.ylabel(r'$\epsilon_{\mathrm{rel}}^*$', fontsize=24)

# Log-scale for y-axis and custom formatting
plt.yscale('log')
#plt.gca().yaxis.set_major_formatter(FuncFormatter(ytick_formatter))

# Ensure gridlines are shown on both axes
plt.grid(visible=True, which='both', linestyle='-', alpha=0.7, linewidth=2)

# Add legend and tighten layout
plt.legend(loc = "upper right", fontsize = 16)
plt.tight_layout()

# Save the plot
output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, 'best_rel_error_vs_diff_steps.png')
plt.savefig(output_path, dpi=1200)
plt.close()
