import sys
sys.path.append("..")
from create_and_run_scripts import create_and_transfer

if(__name__ == "__main__"):
    train_modes = ["REINFORCE", "Forward_KL", "PPO"]
    lrs = [0.0005, 0.00025]
    temps = [0.05]
    n_diffusion_step_list = [ 4,8,16,32,40]
    noise_distributions = ["bernoulli"]  # ["combined", "annealed_obj", "bernoulli"]
    batch_size = 20
    base_diff_steps = 4
    n_GNN_layers = 6
    N_anneal = 1000
    mem_frac = 0.92
    n_basis_states = 5
    for train_mode in train_modes:
        for lr in lrs:
            for temp in temps:
                for n_diffusion_steps in n_diffusion_step_list:
                    for noise_distribution in noise_distributions:

                        if train_mode == "REINFORCE":

                            config = {
                                "CO_problem": "MIS",
                                "dataset": "RB_iid_100",
                                "N_anneal": N_anneal,
                                "lr": lr,
                                'n_GNN_layers': n_GNN_layers,
                                "temp": temp,
                                'noise_potential': noise_distribution,
                                'n_diffusion_steps': n_diffusion_steps,
                                'batch_size': int(base_diff_steps * batch_size / n_diffusion_steps),
                                'n_basis_states': n_basis_states,
                                'train_mode': train_mode,
                                'minib_diff_steps': base_diff_steps,
                                'n_rand_nodes': 3,
                                'diff_schedule': "exp",
                                'seed': 123,
                                'nGPUs': 1,
                                "qos": "default",
                                "mem_frac": mem_frac
                            }
                            create_and_transfer(config)
                        else:
                            config = {
                                "CO_problem": "MIS",
                                "dataset": "RB_iid_100",
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
                                'n_rand_nodes': 3,
                                'diff_schedule': "exp",
                                'seed': 123,
                                'nGPUs': 1,
                                "qos": "default",
                                "mem_frac": mem_frac
                            }
                            create_and_transfer(config)