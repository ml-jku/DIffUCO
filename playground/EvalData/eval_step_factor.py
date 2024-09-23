import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_mean_energy(path_to_models, wandb_run_id, eval_steps, n_states = 8):
    path_folder = f"{path_to_models}/{wandb_run_id}/"

    if not os.path.exists(path_folder):
        print(f"Folder not found: {path_folder}")
        raise ValueError("Folder not found")

    config_file_name = f"best_{wandb_run_id}.pickle"

    with open(os.path.join(path_folder, config_file_name), 'rb') as f:
        test_dict = pickle.load(f)
    config = test_dict[1]
    n_diff_steps = config['n_diffusion_steps']


    eval_step_factors = []
    out_dict = {}
    out_dict["mean_energies"] = []
    out_dict["mean_best_energies"] = []
    out_dict["rel_error_CE"] = []
    out_dict["best_rel_error_CE"] = []

    for eval_step_factor in eval_steps:
        try:
            file_name = f"{wandb_run_id}_test_dict_eval_step_factor_{eval_step_factor}_{n_states}.pickle"

            # Read the pickle file
            with open(os.path.join(path_folder, file_name), 'rb') as f:
                test_dict = pickle.load(f)

            eval_step_factors.append(eval_step_factor)
            out_dict["mean_energies"].append(test_dict["test/mean_energy"])
            out_dict["mean_best_energies"].append(test_dict["test/mean_best_energy"])
            # out_dict["mean_energies"].append(test_dict["test/mean_energy_CE"])
            # out_dict["mean_best_energies"].append(test_dict["test/mean_best_energy_CE"])
            out_dict["rel_error_CE"].append(test_dict["test/rel_error_CE"])
            out_dict["best_rel_error_CE"].append(test_dict["test/mean_best_rel_error_CE"])
        except:
            pass

    return n_diff_steps*np.array(eval_step_factors), out_dict

def plot_over_diff_eval_steps(load_wandb_run_ids):
    import itertools
    markers = itertools.cycle(('d', 'P', 'v', 'o', '*', "<", ">"))
    colors = itertools.cycle(('r', 'g', 'b'))

    path = os.getcwd()
    path_to_models = os.path.dirname(os.path.dirname(path)) + "/Checkpoints"
    eval_step_factors = np.arange(1, 6)
    #wandb_run_id_dict = {"rKL w/o RL": {"wandb_ids": ["08i3m2dl"]}, "rKL w/ RL":  {"wandb_ids": ["fj1lym7o"]} , "fKL w/ MC": {"wandb_ids": ["w3u4cer6"]}}
    wandb_run_id_dict = {}
    for key in load_wandb_run_ids:
        wandb_run_ids = load_wandb_run_ids[key]
        wandb_run_id_dict[key] = {}
        wandb_run_id_dict[key]["eval_step_factors"] = []
        wandb_run_id_dict[key]["mean_energies"] = []
        wandb_run_id_dict[key]["mean_best_energies"] = []
        wandb_run_id_dict[key]["rel_error_CE"] = []
        wandb_run_id_dict[key]["best_rel_error_CE"] = []
        for wandb_run_id in wandb_run_ids:
            n_diff_step_list, out_dict = read_and_plot_mean_energy(path_to_models, wandb_run_id, eval_step_factors)
            wandb_run_id_dict[key]["eval_step_factors"].append(n_diff_step_list)
            wandb_run_id_dict[key]["mean_energies"].append(out_dict["mean_energies"])
            wandb_run_id_dict[key]["mean_best_energies"].append(out_dict["mean_best_energies"])
            wandb_run_id_dict[key]["rel_error_CE"].append(out_dict["rel_error_CE"])
            wandb_run_id_dict[key]["best_rel_error_CE"].append(out_dict["best_rel_error_CE"])

    ms = 14
    lw = 3

    plt.figure(figsize=(10, 6))
    for key in wandb_run_id_dict:
        color = next(colors)
        #marker = next(markers)

        eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
        mean_energies = np.mean(np.array(wandb_run_id_dict[key]["mean_energies"]), axis = 0)
        mean_best_energies = np.mean(np.array(wandb_run_id_dict[key]["mean_best_energies"]), axis = 0)
        plt.plot(eval_step_factors, mean_energies, f'-{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: energy")
        plt.plot(eval_step_factors, mean_best_energies, f'--{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: best energy")
    plt.xlabel('Number of diffusion steps', fontsize=24)
    plt.ylabel('Energy Value', fontsize=24)
    plt.grid(True)
    #plt.axhline(y=-20.10, color='r', linestyle='-', label = "Optimal Energy Value")
    plt.legend(fontsize = 15, loc = "upper right", ncol = 2)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for key in wandb_run_id_dict:
        color = next(colors)

        eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
        mean_energies = np.mean(np.array(wandb_run_id_dict[key]["rel_error_CE"]), axis = 0)
        mean_best_energies = np.mean(np.array(wandb_run_id_dict[key]["best_rel_error_CE"]), axis = 0)
        plt.plot(eval_step_factors, mean_energies, f'-{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: energy")
        plt.plot(eval_step_factors, mean_best_energies, f'--{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: best energy")
    plt.xlabel('Number of diffusion steps', fontsize=24)
    plt.ylabel('Relative Energy', fontsize=24)
    plt.yscale('log')
    # Add both major and minor grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Make horizontal grid lines more visible
    plt.gca().yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize = 15, loc = "upper right", ncol = 2)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.savefig("Relative_energy.png", dpi=1200)
    plt.tight_layout()
    plt.show()


if(__name__ == '__main__'):
    # Usage example
    import itertools
    markers = itertools.cycle(('d', 'P', 'v', 'o', '*', "<", ">"))
    colors = itertools.cycle(('r', 'g', 'b'))

    path = os.getcwd()
    path_to_models = os.path.dirname(os.path.dirname(path)) + "/Checkpoints"
    eval_step_factors = np.arange(1, 6)
    wandb_run_id_dict = {"rKL w/o RL": {"wandb_ids": ["08i3m2dl"]}, "rKL w/ RL":  {"wandb_ids": ["fj1lym7o"]} , "fKL w/ MC": {"wandb_ids": ["w3u4cer6"]}}
    for key in wandb_run_id_dict:
        wandb_run_ids = wandb_run_id_dict[key]["wandb_ids"]
        wandb_run_id_dict[key]["eval_step_factors"] = []
        wandb_run_id_dict[key]["mean_energies"] = []
        wandb_run_id_dict[key]["mean_best_energies"] = []
        wandb_run_id_dict[key]["rel_error_CE"] = []
        wandb_run_id_dict[key]["best_rel_error_CE"] = []
        for wandb_run_id in wandb_run_ids:
            n_diff_step_list, out_dict = read_and_plot_mean_energy(path_to_models, wandb_run_id, eval_step_factors)
            wandb_run_id_dict[key]["eval_step_factors"].append(n_diff_step_list)
            wandb_run_id_dict[key]["mean_energies"].append(out_dict["mean_energies"])
            wandb_run_id_dict[key]["mean_best_energies"].append(out_dict["mean_best_energies"])
            wandb_run_id_dict[key]["rel_error_CE"].append(out_dict["rel_error_CE"])
            wandb_run_id_dict[key]["best_rel_error_CE"].append(out_dict["best_rel_error_CE"])


    # plt.figure(figsize=(10, 6))
    # for key in wandb_run_id_dict:
    #     eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
    #     mean_energies = np.mean(np.array(wandb_run_id_dict[key]["mean_energies"]), axis = 0)
    #     plt.plot(eval_step_factors, mean_energies, '-x', label = key)
    # plt.xlabel('n diffusion steps')
    # plt.ylabel('Mean Energy')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # for key in wandb_run_id_dict:
    #     eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
    #     mean_energies = np.mean(np.array(wandb_run_id_dict[key]["mean_best_energies"]), axis = 0)
    #     plt.plot(eval_step_factors, mean_energies, '-x', label = key)
    # plt.xlabel('n diffusion steps')
    # plt.ylabel('Mean best Energy')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    ms = 14
    lw = 3

    plt.figure(figsize=(10, 6))
    for key in wandb_run_id_dict:
        color = next(colors)
        #marker = next(markers)

        eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
        mean_energies = np.mean(np.array(wandb_run_id_dict[key]["mean_energies"]), axis = 0)
        mean_best_energies = np.mean(np.array(wandb_run_id_dict[key]["mean_best_energies"]), axis = 0)
        plt.plot(eval_step_factors, mean_energies, f'-{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: energy")
        plt.plot(eval_step_factors, mean_best_energies, f'--{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: best energy")
    plt.xlabel('Number of diffusion steps', fontsize=24)
    plt.ylabel('Energy Value', fontsize=24)
    plt.grid(True)
    plt.axhline(y=-20.10, color='r', linestyle='-', label = "Optimal Energy Value")
    plt.legend(fontsize = 15, loc = "upper right", ncol = 2)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for key in wandb_run_id_dict:
        color = next(colors)

        eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
        mean_energies = np.mean(np.array(wandb_run_id_dict[key]["rel_error_CE"]), axis = 0)
        mean_best_energies = np.mean(np.array(wandb_run_id_dict[key]["best_rel_error_CE"]), axis = 0)
        plt.plot(eval_step_factors, mean_energies, f'-{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: energy")
        plt.plot(eval_step_factors, mean_best_energies, f'--{next(markers)}', color = color, markersize = ms, linewidth=lw, label = f"{key}: best energy")
    plt.xlabel('Number of diffusion steps', fontsize=24)
    plt.ylabel('Relative Energy', fontsize=24)
    plt.yscale('log')
    # Add both major and minor grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Make horizontal grid lines more visible
    plt.gca().yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize = 15, loc = "upper right", ncol = 2)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.savefig("Relative_energy.png", dpi=1200)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # for key in wandb_run_id_dict:
    #     eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
    #     mean_energies = np.mean(np.array(wandb_run_id_dict[key]["rel_error_CE"]), axis = 0)
    #     plt.plot(eval_step_factors, mean_energies, '-x', label = key)
    # plt.xlabel('n diffusion steps')
    # plt.ylabel('rel_error_CE')
    # plt.yscale("log")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # for key in wandb_run_id_dict:
    #     eval_step_factors = np.mean(np.array(wandb_run_id_dict[key]["eval_step_factors"]), axis = 0)
    #     mean_energies = np.mean(np.array(wandb_run_id_dict[key]["best_rel_error_CE"]), axis = 0)
    #     plt.plot(eval_step_factors, mean_energies, '--x', label = key)
    # plt.xlabel('n diffusion steps')
    # plt.ylabel('best rel_error_CE')
    # plt.yscale("log")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()