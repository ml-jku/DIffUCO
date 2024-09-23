import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_mean_energy(path_to_models, wandb_run_id, eval_steps, n_states = 8, mode = "test"):
    path_folder = f"{path_to_models}/{wandb_run_id}/"

    if not os.path.exists(path_folder):
        print(f"Folder not found: {path_folder}")
        raise ValueError("Folder not found")

    file_name = f"{wandb_run_id}_test_dict_eval_step_factor_{eval_steps}_{n_states}.pickle"

    # Read the pickle file
    with open(os.path.join(path_folder, file_name), 'rb') as f:
        test_dict = pickle.load(f)


    return test_dict[f"{mode}/energy_mat"], test_dict[f"{mode}/gt_energy_mat"]

if(__name__ == '__main__'):
    # Usage example
    n_states = 150
    path = os.getcwd()
    path_to_models = os.path.dirname(os.path.dirname(path)) + "/Checkpoints"

    wandb_run_id_dict = { "rKL w/ RL":  {"wandb_ids": ["zk3wkaap", "91icd2vu", "fj1lym7o"]}, "rKL w/o RL": {"wandb_ids": ["m3h9mz5g", "olqaqfnl", "08i3m2dl"]} , "fKL w/ MC": {"wandb_ids": ["otpu58r3", "9xoy68e6", "w3u4cer6"]}}
    for key in wandb_run_id_dict:
        wandb_run_ids = wandb_run_id_dict[key]["wandb_ids"]
        wandb_run_id_dict[key]["energy_mat"] = []
        wandb_run_id_dict[key]["gt_energy_mat"] = []
        for wandb_run_id in wandb_run_ids:
            eval_steps = 3
            if(wandb_run_id in wandb_run_id_dict["rKL w/o RL"]["wandb_ids"]):
                eval_steps = 6

            energy_mat, gt_energy_mat = read_and_plot_mean_energy(path_to_models, wandb_run_id, eval_steps, n_states = n_states)
            wandb_run_id_dict[key]["energy_mat"].append(energy_mat)
            wandb_run_id_dict[key]["gt_energy_mat"].append(gt_energy_mat)

    ms = 6
    lw = 2
    import itertools
    markers = itertools.cycle(('d', 'P', 'v', 'o', '*', "<", ">"))
    colors = itertools.cycle(('r', 'g', 'b'))

    plt.figure(figsize=(10, 6))


    for key in wandb_run_id_dict:
        ### TODO add mena over different axes
        color = next(colors)
        marker = next(markers)

        print("data shape", np.array(wandb_run_id_dict[key]["energy_mat"]).shape)
        n_seeds = np.array(wandb_run_id_dict[key]["energy_mat"]).shape[0]
        mean_energy_over_states = np.array([np.mean(np.min(np.array(wandb_run_id_dict[key]["energy_mat"])[:,:,0:i+1,0], axis = -1)) for i in range(n_states - 1)])
        mean_energy_over_states_err = np.array([np.std(np.mean(np.min(np.array(wandb_run_id_dict[key]["energy_mat"])[:,:,0:i+1,0], axis = -1), axis = -1))/np.sqrt(n_seeds) for i in range(n_states - 1)])
        plt.fill_between(np.arange(1, n_states), mean_energy_over_states - mean_energy_over_states_err, mean_energy_over_states + mean_energy_over_states_err, color = color, linewidth = lw, alpha = 0.5)

        plt.errorbar(np.arange(1, n_states), mean_energy_over_states, yerr=mean_energy_over_states_err, fmt = f'-{marker}', color = color, markersize = ms, linewidth = lw, label = key+ ": best value", alpha = 0.3)

        mean_energy_over_states = np.array([np.mean(np.mean(np.array(wandb_run_id_dict[key]["energy_mat"])[...,0], axis = -1)) for i in range(n_states - 1)])
        mean_energy_over_states_err = np.array([np.std(np.mean(np.mean(np.array(wandb_run_id_dict[key]["energy_mat"])[...,0], axis = -1), axis = -1))/np.sqrt(n_seeds) for i in range(n_states - 1)])
        plt.fill_between(np.arange(1, n_states), mean_energy_over_states - mean_energy_over_states_err,
                         mean_energy_over_states + mean_energy_over_states_err, color=color, linewidth=lw,
                         alpha=0.5)

        plt.errorbar(np.arange(1, n_states), mean_energy_over_states, yerr=mean_energy_over_states_err,
                     fmt=f'-{marker}', color=color, markersize=ms, linewidth=lw, label=key + ": average", alpha=0.3)

    plt.xlabel('Number of States',fontsize = 24)
    plt.ylabel('Energy Value', fontsize = 24)
    plt.axhline(y=-20.10, color='r', linestyle='--', linewidth = lw, label = "Optimal Energy Value")
    plt.grid(True)
    plt.legend(fontsize = 16, loc = "upper center", ncol = 2)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.tight_layout()
    plt.savefig("Energy_over_states.png", dpi=1200)
    plt.show()