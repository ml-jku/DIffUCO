import sys
sys.path.append("..")
from create_and_run_scripts import create_and_transfer

if(__name__ == "__main__"):

    dataset = "BA_large"
    CO_problem = "MDS"
    train_modes = ["PPO"]
    lrs = [0.0005]
    temps = [0.1]
    seeds = [ 123, 124]
    n_diffusion_step_list = [ 12]
    noise_distributions = ["bernoulli"]  # ["combined", "annealed_obj", "bernoulli"]
    batch_size = 60
    base_diff_steps = 6
    n_basis_states = 4
    mem_frac = 0.92
    N_anneal = 2000
    nGPUs = 2
    n_GNN_layers = 6
    proj_name = "final_runs"
    mov_average = 0.09
    n_rand_nodes = 1

    config_list = []

    for seed in seeds:
        for train_mode in train_modes:
            for lr in lrs:
                for temp in temps:
                    for n_diffusion_steps in n_diffusion_step_list:
                        for noise_distribution in noise_distributions:

                                config = {
                                    "CO_problem": CO_problem,
                                    "dataset": dataset,
                                    "N_anneal": N_anneal,
                                    "lr": lr,
                                    'n_GNN_layers': n_GNN_layers,
                                    "temp": temp,
                                    'noise_potential': noise_distribution,
                                    'n_diffusion_steps': n_diffusion_steps,
                                    'batch_size': batch_size,  # int(3*batch_size/n_diffusion_steps),
                                    'n_basis_states': n_basis_states,
                                    'train_mode': train_mode,
                                    'minib_diff_steps': base_diff_steps,
                                    'n_rand_nodes': n_rand_nodes,
                                    'diff_schedule': "exp",
                                    'seed': seed,
                                    'nGPUs': nGPUs,
                                    "qos": "default",
                                    "mem_frac": mem_frac,
                                    "mov_average": mov_average,
                                }
                                config_list.append(config)

    # dataset = "BA_small"
    # CO_problem = "MaxCut"
    # train_modes = ["PPO"]
    # lrs = [0.0005]
    # temps = [0.5]
    # seeds = [ 124, 125]
    # n_diffusion_step_list = [ 14]
    # noise_distributions = ["bernoulli"]  # ["combined", "annealed_obj", "bernoulli"]
    # batch_size = 140
    # base_diff_steps = 7
    # n_basis_states = 4
    # mem_frac = 0.92
    # N_anneal = 2000
    # nGPUs = 1
    # n_GNN_layers = 8
    # proj_name = "final_runs"
    #
    #
    # for seed in seeds:
    #     for train_mode in train_modes:
    #         for lr in lrs:
    #             for temp in temps:
    #                 for n_diffusion_steps in n_diffusion_step_list:
    #                     for noise_distribution in noise_distributions:
    #
    #                             config = {
    #                                 "CO_problem": CO_problem,
    #                                 "dataset": dataset,
    #                                 "N_anneal": N_anneal,
    #                                 "lr": lr,
    #                                 'n_GNN_layers': n_GNN_layers,
    #                                 "temp": temp,
    #                                 'noise_potential': noise_distribution,
    #                                 'n_diffusion_steps': n_diffusion_steps,
    #                                 'batch_size': batch_size,  # int(3*batch_size/n_diffusion_steps),
    #                                 'n_basis_states': n_basis_states,
    #                                 'train_mode': train_mode,
    #                                 'minib_diff_steps': base_diff_steps,
    #                                 'n_rand_nodes': 3,
    #                                 'diff_schedule': "exp",
    #                                 'seed': seed,
    #                                 'nGPUs': nGPUs,
    #                                 "qos": "default",
    #                                 "mem_frac": mem_frac,
    #                                 "mov_average": 0.09
    #                             }
    #                             config_list.append(config)

    create_and_transfer(config_list, proj_name = proj_name)