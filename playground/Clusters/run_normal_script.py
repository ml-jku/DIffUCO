import sys
sys.path.append("..")
from create_and_run_scripts import create_and_transfer

if(__name__ == "__main__"):
    #python argparse_ray_main.py --lrs 0.0005 --relaxed --GPUs 0 1 --n_GNN_layers 6 --temps 0.1 --IsingMode RB_iid_small --EnergyFunction MIS --mode Diffusion --N_anneal 2000 --beta_factor 1. --n_diffusion_steps 12 --batch_size 30 --n_basis_states 5 --noise_potential bernoulli --multi_gpu --project_name scaling_law --n_rand_nodes 3 --seed 123 --graph_mode normal --train_mode PPO --inner_loop_steps 1 --diff_schedule exp --minib_diff_steps 4 --minib_basis_states 5

#python argparse_ray_main.py --lrs 0.0005 --relaxed --GPUs 0 1 --n_GNN_layers 6 --temps 0.1 --IsingMode RB_iid_small --EnergyFunction MIS --mode Diffusion --N_anneal 2000 --beta_factor 1. --n_diffusion_steps 16 --batch_size 30 --n_basis_states 5 --noise_potential bernoulli --multi_gpu --project_name scaling_law --n_rand_nodes 3 --seed 123 --graph_mode normal --train_mode PPO --inner_loop_steps 1 --diff_schedule exp --minib_diff_steps 4 --minib_basis_states 5


    dataset = "RB_iid_small"
    CO_problem = "MIS"
    train_modes = ["PPO"]
    lrs = [0.0005]
    temps = [0.1]
    seeds = [ 123]
    n_diffusion_step_list = [ 12, 16]
    noise_distributions = ["bernoulli"]  # ["combined", "annealed_obj", "bernoulli"]
    batch_size = 30
    base_diff_steps = 4
    n_basis_states = 5
    mem_frac = 0.92
    N_anneal = 2000
    nGPUs = 1
    n_GNN_layers = 6
    proj_name = "scaling_law"
    mov_average = 0.0009
    n_rand_nodes = 3

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

    dataset = "RB_iid_small"
    CO_problem = "MIS"
    train_modes = ["Forward_KL"]
    lrs = [0.0005]
    temps = [0.1]
    seeds = [ 123]
    n_diffusion_step_list = [ 12, 16]
    noise_distributions = ["bernoulli"]  # ["combined", "annealed_obj", "bernoulli"]
    batch_size = 30
    base_diff_steps = 4
    n_basis_states = 5
    mem_frac = 0.92
    N_anneal = 2000
    nGPUs = 1
    n_GNN_layers = 6
    proj_name = "scaling_law"
    n_rand_nodes = 3
    
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
                                    "mov_average": 0.09
                                }
                                config_list.append(config)

    create_and_transfer(config_list, proj_name = proj_name)