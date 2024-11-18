import math
from functools import partial

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state
import numpy as np
import os
# import sys
# sys.path.append("..")
import os

from Network.AutoregressiveNN import AutoregressiveNN
from Network.AutoregressiveGraphNN import AutoregressiveGraphNN, AutoregressiveTrainer
#from Network.AutoregressiveGraphNN import AutoregressiveTrainer
from ImportanceSampler.ImportanceSamplerClass import calc_log_w_hat, free_energy
from Jraph_creator.JraphCreator import create_graph
from Energies.energy import energy_registry
from ising_free_energy import calculate_ising_free_energy_exact

import argparse
import jax.numpy as jnp

parser = argparse.ArgumentParser()

parser.add_argument('--size', default=4, type = int)
parser.add_argument('--seed', default=123, type = int)
parser.add_argument('--epochs', default=10000, type = int)
parser.add_argument('--GPU', default="0", type = str)
parser.add_argument('--lr', default=0.001, type = float)
parser.add_argument('--beta_target', default=0.4407, type = float)
parser.add_argument('--T_start', default=4., type = float)
parser.add_argument('--n_layers', default=1, type = int)

parser.add_argument('--nh_MLP', default=16, type = int)
parser.add_argument('--nh_conv', default=16, type = int)
parser.add_argument('--n_samples', default=50, type = int, help = "number of samples for each graph")
parser.add_argument('--lam', default=5, type = int, help = "scaling for anneal schedule")
parser.add_argument('--eval_n_samples', default=250, type = int, help = "number of samples for each graph")
parser.add_argument("--energy_func", default="Ising", type = str, choices=["Ising", "SpinGlass"], help = "energy function to use")

args = parser.parse_args()

hamiltonian = energy_registry(args.size, energy_func = args.energy_func)

@partial(jax.jit, static_argnums=())
def REINFORCE_loss(params, H_Graph, T, prev_sample_arr, sample):
    log_prob_per_state = vmap_compute_log_likelihood_of_sample(params, prev_sample_arr, sample)
    #log_prob_per_state = ann.log_likelihood( sample, log_probs)

    Energy = hamiltonian(H_Graph, sample)
    R = T * log_prob_per_state + Energy
    loss = jax.lax.stop_gradient(R)

    loss_reinforce = ((loss - loss.mean()) * log_prob_per_state).mean()
    return loss_reinforce, (Energy, log_prob_per_state)

@partial(jax.jit, static_argnums=(0, 1))
def train_step( num_spins, batch_size,T, H_Graph, state, key):
    sample, log_probs, prev_sample_arr, next_sample_arr = ann.generate_sample(num_spins, batch_size, state.params, key, 0)
    #print("sample shape", sample.shape)
    (loss, (Energy, log_prob_per_state)), grads = grad_fn( state.params, H_Graph, T, prev_sample_arr, sample)
    state = state.apply_gradients(grads=grads)
    return state, loss, (Energy, log_prob_per_state)

def cos_schedule(epoch, N_anneal, max_lr = 10**-3, min_lr = 10**-4, f_warmup = 0.025):
	start_lr = 10**-10
	new_lr = jnp.where(epoch < N_anneal*f_warmup, start_lr + (epoch)/(N_anneal*f_warmup)*(max_lr - start_lr), (max_lr-min_lr)*jnp.cos(jnp.pi/N_anneal * epoch)/2 + min_lr + (max_lr-min_lr)/2)
	new_lr = jnp.where(epoch > N_anneal , min_lr, new_lr)
	return new_lr

def create_train_state(model , H_graph, key, learning_rate, n, epochs, warmup_frac = 0.025):
    key, subkey = jax.random.split(key)

    params = model.init({"params": subkey}, jnp.ones((1,n, 1)))

    lr_schedule = lambda step: -cos_schedule(step, epochs, max_lr=learning_rate, min_lr=learning_rate / 10)
    tx = optax.chain( optax.scale_by_radam(),
                optax.scale_by_schedule(lr_schedule))
    print("parameters")
    print(jax.tree_util.tree_map(lambda x: x.shape, params))
    print(jax.tree_util.tree_map(lambda x: x.dtype, params))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _init_wandb(config, id, project = ""):
    """
    initialize weights and biases

    @param project: project name
    """
    if config["wandb"] is True:
        wandb.init(project=project, name=f"{args.energy_func}_{id}_T_init_{config['T_init']}", group=id, id=id,
                   config=config, mode="online", settings=wandb.Settings(_service_wait=300))


def anneal_temp(initial_temp, final_temp, num_epochs, epoch):
    return final_temp * 1 / (1 - 0.998 ** (args.lam/100 * (epoch + 1)))

def test_magnetization(ann, params, num_spins, batch_size, key):
    sample, x_hat, _ , _ = ann.generate_sample(num_spins, batch_size, params, key, 0)
    abs_mean_M =  np.mean(np.abs(np.sum(sample, axis = 1)))
    mean_M = np.mean(np.sum(sample, axis = 1))
    wandb.log({"mean_abs_M": float(abs_mean_M), "mean_M": float(mean_M)})


@partial(jax.jit, static_argnums=(0,))
def calculate_free_energy(ann, params, temp, H_Graph):
    sample, log_probs, _ , _ = ann.generate_sample(num_spins, eval_batch_size, params, key, 0)
    loglikelihood_vals = ann.log_likelihood(sample, log_probs)

    log_w_hat = calc_log_w_hat(loglikelihood_vals, temp, H_Graph, sample, hamiltonian)
    O_F = free_energy(1 / target_temp, log_w_hat, int(math.sqrt(num_spins)))
    return O_F

if __name__ == "__main__":
    ### Diffusion
    diff_epochs =  450
    diff_n_states = 100
    diff_steps = 300
    diff_graphs_per_epoch = 100
    AR_memory_forw_passes = args.size**2
    diff_memory_forw_passes = 100


    batch_size = diff_n_states*AR_memory_forw_passes/diff_memory_forw_passes
    print("batch size should be", batch_size)

    ### TODO implement lr schedule
    epochs = int(diff_graphs_per_epoch* diff_epochs*diff_n_states)

    from jax import config
    wandb_id = wandb.util.generate_id()
    #config.update('jax_disable_jit', True)
    device = args.GPU
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.96"

    n = args.size

    seed = args.seed
    key = jax.random.key(seed)

    num_spins = n * n
    learning_rate = args.lr
    batch_size = args.n_samples
    eval_batch_size = args.eval_n_samples

    initial_temp = args.T_start
    current_temp = initial_temp
    target_temp = 1 / args.beta_target

    wandb_config = {
        "Ising_size": n,
        "seed": seed,
        "lr": learning_rate,
        "batch_size": batch_size,
        "T_init": initial_temp,
        "target_temp": target_temp,
        "wandb": True,
        "epochs": args.epochs,
        "nh_conv": args.nh_conv,
        "nh_MLP": args.nh_MLP,
        "n_layers": args.n_layers
    }

    print("config", config)

    _init_wandb(wandb_config, wandb_id, project = f"AR_Ising_{n}x{n}")
    wandb_run_id = wandb.run.id
    print("Starting run with run id", wandb_run_id)
    H_Graph = create_graph(int(jnp.sqrt(num_spins)))
    if(args.energy_func == "Ising"):
        ann = AutoregressiveNN(grid_size = n, n_layers = args.n_layers, features = args.nh_MLP, cnn_features = args.nh_conv)
    else:
        np.random.seed(0)
        half_edges = np.random.normal(0,1, (H_Graph.senders.shape[0]//2,1))
        edges = np.concatenate([half_edges, half_edges], axis = -1)
        edges = np.ravel(edges)[:, None]
        H_Graph = H_Graph._replace(edges = jnp.array(edges))
        ann = AutoregressiveGraphNN(n_message_passes = args.n_layers, nh = args.nh_MLP)
    vmap_compute_log_likelihood_of_sample = jax.vmap(ann.compute_log_likelihood_of_sample, in_axes=(None, 1, 0),
                                                     out_axes=(0))
    grad_fn = jax.value_and_grad(lambda a, b, c, d, e: REINFORCE_loss(a, b, c, d, e), has_aux=True)
    #ann = AutoregressiveGraphNN()
    #ann_trainer = AutoregressiveTrainer(ann, batch_size = batch_size)
    # create graph
    # Create train state
    state = create_train_state(ann, H_Graph, key, learning_rate, num_spins, epochs)
    epochs = args.epochs
    O_F_exact = calculate_ising_free_energy_exact(1 / target_temp, n)
    for epoch in range(epochs):
        current_temp = anneal_temp(initial_temp, target_temp, epochs, epoch)
        key, subkey = jax.random.split(key)
        state, loss, (energy, _) = train_step(num_spins, batch_size, current_temp, H_Graph, state, subkey)

        wandb.log({"loss": loss})
        wandb.log({"T": float(current_temp)})
        wandb.log({"Energy": float(energy.mean())})
        print("curr epoch is", epoch, loss)

        if epoch % 100 == 0:
            # test for temperature
            #config.update("jax_enable_x64", True)
            test_magnetization(ann, state.params, num_spins, eval_batch_size, subkey)
            ### TODO calculate effective samples size
            ### TODO make seversal sampling round
            O_F = calculate_free_energy(ann, state.params, target_temp, H_Graph)
            wandb.log({"Free Energy Diff": abs(O_F - O_F_exact)})
            wandb.log({"Free Energy": float(O_F)})
            #config.update("jax_enable_x64", False)

    AutoregressiveNN.save_params(state.params, wandb_config, wandb_id)

    sample, x_hat = ann.generate_sample(num_spins, 5, state.params, key, 0)
    print("Single sample:")
    print(sample)
