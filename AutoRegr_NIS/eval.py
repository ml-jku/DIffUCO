import os
import pickle

import math

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import argparse
from Network.AutoregressiveNN import AutoregressiveNN
from Jraph_creator.JraphCreator import create_graph
from ImportanceSampler import ImportanceSamplerClass as IS
import wandb

parser = argparse.ArgumentParser()

parser.add_argument('--wandb_ids', default=["fi3b7s13"], type = str, nargs = "+")
parser.add_argument('--GPU', default="3", type = str)
parser.add_argument('--batch_size', default=1400, type = int)
parser.add_argument('--iterations', default=400, type = int)
parser.add_argument('--epsilons', default=[0.], type = str, nargs = "+")
parser.add_argument('--seeds', default=3, type = int)

args = parser.parse_args()


if(__name__ == "__main__"):
    if __name__ == "__main__":
        device = args.GPU
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.96"
        ## apply importance sampling
        from jax import config as jax_config
        #jax_config.update("jax_enable_x64", True)
        jax.config.update('jax_platform_name', "cpu")
        # TODO add effecitve sample size

        for wandb_id in args.wandb_ids:
            ann = AutoregressiveNN(grid_size = 1)
            params, config = ann.load_params(wandb_id)

            wandb.init(project="eval_AR_Ising_models", name=f"{id}_T_init_{config['T_init']}", group=f"eval_{wandb_id}", id=f"eval_{wandb_id}",
                       config=config, mode="online", settings=wandb.Settings(_service_wait=300))

            seed = config["seed"]
            key = jax.random.key(seed)
            n = config["Ising_size"]
            num_spins = n * n
            target_temp = config["target_temp"]
            H_Graph = create_graph(int(jnp.sqrt(num_spins)))
            ann = AutoregressiveNN(grid_size = n, n_layers = config["n_layers"], features = config["nh_MLP"], cnn_features = config["nh_conv"])

            importance = IS.ImportanceSampler()

            batch_size = args.batch_size
            iterations = args.iterations

            epsilon = args.epsilons

            fig, ax = plt.subplots(1, 1)
            cmap = mpl.colormaps['plasma']
            colors = cmap(np.linspace(0, 1, len(epsilon)))
            x = np.arange(0,iterations)

            for seed in range(args.seeds):
                for i, eps in enumerate(epsilon):
                    O_F_diffs = []
                    O_F_diffs_p = []
                    O_F_diffs_m = []

                    out_dict_lists = {"Free_Energy": [], "Free_Energy_lower_bound": [], "Free_Energy_upper_bound": [],
                                "Free_Energy_exact": [],
                                "Entropy": [], "Entropy_exact": [], "Inner_Energy": [],
                                "Inner_Energy_exact": [], "effective_sample_size": []}

                    sample_acc = None
                    log_probs_acc = None
                    for rep in range(iterations):
                        print(rep)
                        key, subkey = jax.random.split(key)
                        sample, log_probs, _ , _ = ann.generate_sample(num_spins, batch_size, params, subkey, eps)

                        if sample_acc is not None:
                            sample_acc = jnp.concat((sample_acc, sample))
                            log_probs_acc = jnp.concat((log_probs_acc, log_probs))
                        else:
                            sample_acc = sample
                            log_probs_acc = log_probs

                        out_dict = importance.run(H_Graph, target_temp, num_spins, ann, sample_acc, log_probs_acc)
                        O_F_diffs.append(float(np.abs(out_dict["Free_Energy"] - out_dict["Free_Energy_exact"])))
                        O_F_diffs_p.append(float(np.abs(out_dict["Free_Energy_lower_bound"] - out_dict["Free_Energy_exact"])))
                        O_F_diffs_m.append(float(np.abs(out_dict["Free_Energy_upper_bound"] - out_dict["Free_Energy_exact"])))

                        for dict_key in out_dict.keys():
                            out_dict_lists[dict_key].append(out_dict[dict_key])

                    ann.save_dict(out_dict_lists, wandb_id, seed)

                    ax.plot(x, O_F_diffs, label=str(eps), color=colors[i])
                    plt.fill_between(x, O_F_diffs_m, O_F_diffs_p, color=colors[i], alpha=0.1)

                plt.axhline(y=0.0, color='r', linestyle='-')
                plt.legend()

                wandb.log({"Free_Energy_fig": wandb.Image(fig)})
                plt.close()



            ### TODO save with picke and log also entropy and Energy and effective sample size