import sys
sys.path.append("..")
from create_and_run_scripts import create_and_transfer

if(__name__ == "__main__"):
    ### TODO use whole node by running several jobs per node

    dataset = "NxNLattice_8x8"
    CO_problem = "IsingModel"
    train_modes = ["Forward_KL"]
    lrs = [0.0001]
    temps = [6.]
    seeds = [ 123]
    n_diffusion_step_list = [ 16, 32, 64, 128]
    noise_distributions = ["bernoulli"]  # ["combined", "annealed_obj", "bernoulli"]
    batch_size = 1
    base_diff_steps = 8
    n_basis_states = 80
    mem_frac = 0.92
    max_N_anneal = 2000*100
    nGPUs = 1
    n_GNN_layers = 8
    proj_name = "Forward_KL_"

    config_list = []

    for seed in seeds:
        for train_mode in train_modes:
            for lr in lrs:
                for temp in temps:
                    for n_diffusion_steps in n_diffusion_step_list:
                        for noise_distribution in noise_distributions:
                                N_anneal = int(max_N_anneal/n_diffusion_steps)
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
                                    'n_rand_nodes': 0,
                                    'diff_schedule': "exp",
                                    'seed': seed,
                                    'nGPUs': nGPUs,
                                    "qos": "default",
                                    "mem_frac": mem_frac,
                                    "T_target": 2.26911731337,
                                    "n_sampling_rounds": 100,
                                    "n_test_basis_states": 300,
                                    "minib_basis_states": 80,
                                    "n_hidden_neurons": 128
                                }
                                config_list.append(config)


    create_and_transfer(config_list, proj_name = proj_name)