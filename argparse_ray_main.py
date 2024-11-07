import os
import argparse
from train import TrainMeanField


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Switch ray into local mode for debugging')
parser.add_argument('--multi_gpu', action='store_true', help='wheter to use multi gpu or not, KEEP IT ALWAYS TRUE')
parser.add_argument('--mode', default='Diffusion', choices = ["Diffusion"], help='Define the Approach')
parser.add_argument('--EnergyFunction', default='MIS', choices = ["MaxCut", "MIS", "MVC", "MaxCl", "WMIS", "MDS", "MaxClv2", "TSP", "IsingModel"], help='Define the EnergyFunction of the IsingModel')
parser.add_argument('--IsingMode', default='RB_iid_100', choices = ["Gset","BA_large","RB_iid_small", "RB_iid_dummy", "BA_dummy", "RB_iid_large" ,"RRG_200_k_=all", "BA_small","TSP_random_100", "TSP_random_20", "COLLAB", "IMDB-BINARY", "RB_iid_100_dummy" , "RB_iid_200", "RB_iid_100", "NxNLattice_4x4", "NxNLattice_8x8", "NxNLattice_16x16", "NxNLattice_24x24", "NxNLattice_32x32"], help='Define the Training dataset')
parser.add_argument('--graph_mode', default='normal', choices = ["normal", "TSPModel", "Transformer", "UNet"], help='Use U-Net or normal GNN')
parser.add_argument('--train_mode', default='REINFORCE', choices = ["REINFORCE", "PPO", "Forward_KL"], help='Use U-Net or normal GNN')
parser.add_argument('--AnnealSchedule', default='linear', choices = ["linear", "cosine", "exp"], help='Define the Annealing Schedule')
parser.add_argument('--temps', default=[0.], type = float, help='Define gridsearch over Temperature', nargs = "+")
parser.add_argument('--T_target', default=0., type = float, help='Define target temperature')
parser.add_argument('--N_warmup', default=0, type = int, help='Define gridsearch over Number of Annealing steps')
parser.add_argument('--N_anneal', default=[2000], type = int, help='Define gridsearch over Number of Annealing steps', nargs = "+")
parser.add_argument('--N_equil', default = 0, type = int, help='Define gridsearch over Number of Equil steps')
parser.add_argument('--lrs', default=[5e-5], type = float, help='Define gridsearch over learning rate', nargs = "+")
parser.add_argument('--lr_schedule', default="cosine", choices = ["cosine", "None"], help='use learning rate schedule or not')
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--GPUs', default=["0"], type = str, help='Define Nb', nargs = "+")
parser.add_argument('--n_hidden_neurons', default=[64], type = int, help='number of hidden neurons', nargs = "+")
parser.add_argument('--n_rand_nodes', default=2, type = int, help='define node embedding size')
parser.add_argument('--stop_epochs', default=10000, type = int, help='define early stopping')
parser.add_argument('--n_diffusion_steps', default=[9], type = int, help='define number of diffusion steps', nargs = "+")
parser.add_argument('--time_encoding', default="one_hot", type = str, help='encoding of diffusion steps')
parser.add_argument('--noise_potential', default = ["annealed_obj"], type = str, choices = ["bernoulli", "boltzmann_noise", "diffusion", "annealed_obj", "categorical", "combined"], help='define the diffusion mode', nargs = "+")
parser.add_argument('--n_basis_states', default=[10], type = int, help='number of states per graph', nargs = "+")
parser.add_argument('--n_test_basis_states', default=8, type = int, help='number of states per graph during test time')
parser.add_argument('--batch_size', default=[30], type = int, help='number of graphs within a batch', nargs = "+")
parser.add_argument('--minib_diff_steps', default=1, type = int, help='minibatch size in diffusion steps in PPO or forward KL')
parser.add_argument('--minib_basis_states', default=10, type = int, help='minibatch size in basis states in PPO or forward KL')
parser.add_argument('--inner_loop_steps', default=1, type = int, help='number of inner loop steps in PPO or forward KL')
parser.add_argument('--n_GNN_layers', default=[8], type = int, help='num of GNN Layers', nargs = "+")
parser.add_argument('--project_name', default= "", type = str, help='define project name')
parser.add_argument('--beta_factor', default=[1.], type = float, help='desfine noise strength', nargs = "+")
parser.add_argument('--loss_alpha', default=0.0, type = float, help='rel weighteing between forward and reverse KL')
parser.add_argument('--MCMC_steps', default=0, type = int, help='number of MCMC steps')
parser.add_argument('--mov_average', default=0.0009, type = float, help='moving_average for RL')
parser.add_argument('--TD_k', default=3, type = float, help='TD_k for PPO')
parser.add_argument('--clip_value', default=0.2, type = float, help='clip_value for PPO')
parser.add_argument('--value_weighting', default=0.65, type = float, help='value_func weighting for PPO')
parser.add_argument('--mem_frac', default= ".90", type = str, help='memory fraction')
parser.add_argument('--diff_schedule', default= "own", type = str, help='define diffusion schedule')
parser.add_argument('--proj_method', default= "None", choices = ["CE", "feasible", "None"], type = str, help='define projection method')
parser.add_argument('--linear_message_passing', action='store_true')
parser.add_argument('--no-linear_message_passing', dest='linear_message_passing', action='store_false')
parser.add_argument('--relaxed', action='store_true')
parser.add_argument('--no-relaxed', dest='relaxed', action='store_false')
parser.add_argument('--time_conditioning', action='store_true')
parser.add_argument('--no-time_conditioning', dest='time_conditioning', action='store_false')
parser.add_argument('--deallocate', action='store_true')
parser.add_argument('--no-deallocate', dest='time_conditioning', action='store_false')
parser.add_argument('--jit', action='store_true')
parser.add_argument('--no-jit', dest='jit', action='store_false')
parser.add_argument('--mean_aggr', action='store_true')
parser.add_argument('--no-mean_aggr', dest='mean_aggr', action='store_false')
parser.add_argument('--grad_clip', action='store_true')
parser.add_argument('--no-grad_clip', dest='grad_clip', action='store_false')
parser.add_argument('--graph_norm', action='store_true')
parser.add_argument('--no-graph_norm', dest='graph_norm', action='store_false')
parser.add_argument('--sampling-temp', default=0., type = float, help='define sampling temperature for asymptoticly unbiased estimations')
parser.add_argument('--n_sampling_rounds', default=5, type = int, help='how often the the basis states are sampled in a loop in unbiased estimations')
parser.add_argument('--bfloat16', action='store_true')
parser.add_argument('--no-bfloat16', dest='bfloat16', action='store_false')
parser.set_defaults(bfloat16=False)

parser.set_defaults(CE=False)
parser.set_defaults(graph_norm=True)
parser.set_defaults(grad_clip=True)
parser.set_defaults(mean_aggr=True)
parser.set_defaults(relaxed=True)
parser.set_defaults(time_conditioning=True)
parser.set_defaults(deallocate=False)
parser.set_defaults(jit=True)
parser.set_defaults(linear_message_passing=True)
parser.set_defaults(multi_gpu=True)
args = parser.parse_args()

### TODO add MaxCut

### TODO moving average mean should also be saved in checkpoint
### TODO add clip stuff and so on into config!
### TODO rerun checkpoint from best checkpoint
def meanfield_run():



    resources_per_trial = 1.
    devices = args.GPUs
    n_workers = int(len(devices)/resources_per_trial)

    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str)

    if(len(args.GPUs) > 1):
        device_str = ""
        for idx, device in enumerate(devices):
            if (idx != len(devices) - 1):
                device_str += str(devices[idx]) + ","
            else:
                device_str += str(devices[idx])

        print(device_str, type(device_str))
    else:
        device_str = str(args.GPUs[0])

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str

    nh = args.n_hidden_neurons[0]

    local_mode = args.debug

    if local_mode:
        print("Init ray in local_mode!")
    elif(args.multi_gpu):
        pass

    #run_PPO_experiment_func = lambda flex_conf: run_PPO_experiment_hydra()
    if(local_mode):
        import jax
        #jax.config.update('jax_platform_name', 'cpu')
        run(flexible_config = {"jit": False}, overwrite = True)
    elif(args.multi_gpu):
        detect_and_run_for_loops()
    # else:
    #     os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    #     from datetime import datetime
    #     # datetime object containing current date and time
    #     now = datetime.now()
    #     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #
    #     trainable_with_gpu = tune.with_resources(run, {"gpu": 1})
    #     tuner = tune.Tuner(trainable_with_gpu, param_space=flexible_config,
    #                        run_config=ray.air.RunConfig(storage_path=f"{os.getcwd()}/ray_results",
    #                                                     name=f"test_experiment_{dt_string}"))
    #
    #     results = tuner.fit()

def detect_and_run_for_loops():
    nh = args.n_hidden_neurons[0]

    seeds = args.seed
    lrs = args.lrs
    N_anneals = args.N_anneal
    GNN_layers = args.n_GNN_layers
    temps = args.temps
    n_diffusion_steps = args.n_diffusion_steps
    n_basis_states = args.n_basis_states
    beta_factors = args.beta_factor
    batch_sizes = args.batch_size
    noise_potentials = args.noise_potential

    for seed in seeds:
        for lr in lrs:
            for N_anneal in N_anneals:
                for GNN_layer in GNN_layers:
                    for temp in temps:
                        for diff_steps in n_diffusion_steps:
                            for n_basis_state in n_basis_states:
                                for beta_factor in beta_factors:
                                    for batch_size in batch_sizes:
                                        for noise_potential in noise_potentials:

                                            ###checks
                                            if(args.train_mode != "REINFORCE"):
                                                if(diff_steps%args.minib_diff_steps!= 0):
                                                    raise ValueError("args.n_diffusion_steps%args.miniminib_diff_steps is not zero!")
                                                if(n_basis_state%args.minib_basis_states!= 0):
                                                    raise ValueError("args.n_basis_sates%args.minib_basis_states is not zero!")

                                                if (batch_size % len(args.GPUs) != 0):
                                                    raise ValueError("args.batch_size%len(args.GPUs) should be zero!")

                                            flexible_config = {
                                                "mode": args.mode,
                                                "dataset_name": args.IsingMode,
                                                "problem_name": args.EnergyFunction,
                                                "jit": args.jit,
                                                "wandb": True,

                                                "seed": seed,
                                                "lr": lr,

                                                "random_node_features": True,
                                                "n_random_node_features": args.n_rand_nodes,
                                                "relaxed": args.relaxed,
                                                "T_max": temp,
                                                "N_warmup": args.N_warmup,
                                                "N_anneal": N_anneal,
                                                "N_equil": args.N_equil,
                                                "n_hidden_neurons": nh,
                                                "n_features_list_prob": [2],
                                                "n_features_list_nodes": [nh, nh],
                                                "n_features_list_edges": [nh, nh],
                                                "n_features_list_messages": [nh, nh],
                                                "n_features_list_encode": [nh, nh],
                                                "n_features_list_decode": [nh, nh],
                                                "n_message_passes": GNN_layer,
                                                "message_passing_weight_tied": False,
                                                "n_diffusion_steps": diff_steps,
                                                "N_basis_states": n_basis_state,
                                                "batch_size": batch_size,
                                                "beta_factor": beta_factor,
                                                "stop_epochs": args.stop_epochs,
                                                "noise_potential": noise_potential,
                                                "time_conditioning": args.time_conditioning,
                                                "project_name": args.project_name,
                                                "linear_message_passing": args.linear_message_passing,

                                                "n_random_node_features": args.n_rand_nodes,
                                                "mean_aggr": args.mean_aggr,
                                                "grad_clip": args.grad_clip,
                                                "graph_mode": args.graph_mode,
                                                "loss_alpha": args.loss_alpha,
                                                "MCMC_steps": args.MCMC_steps,
                                                "train_mode": args.train_mode,

                                                "inner_loop_steps": args.inner_loop_steps,
                                                "minib_diff_steps": args.minib_diff_steps,
                                                "minib_basis_states": args.minib_basis_states,
                                                "graph_norm": args.graph_norm,
                                                "proj_method": args.proj_method,
                                                "diff_schedule": args.diff_schedule,
                                                "mov_average": args.mov_average,
                                                "sampling_temp": args.sampling_temp,
                                                "n_sampling_rounds": args.n_sampling_rounds,
                                                "n_test_basis_states": args.n_test_basis_states,
                                                "bfloat16": args.bfloat16,
                                                "T_target": args.T_target,
                                                "AnnealSchedule": args.AnnealSchedule,
                                                "time_encoding": args.time_encoding,
                                                "lr_schedule": args.lr_schedule,
                                                "TD_k": args.TD_k,
                                                "clip_value": args.clip_value,
                                                "value_weighting": args.value_weighting
                                            }

                                            run(flexible_config=flexible_config, overwrite=True)




def run( flexible_config, overwrite = True):

    config = {
        "mode": "Diffusion",  # either Diffusion or MeanField
        "dataset_name": "RB_iid_small",
        "problem_name": "MIS",
        "jit": True,
        "wandb": True,

        "seed": 123,
        "lr": 1e-4,
        "batch_size": 30, # H
        "N_basis_states": 30, # n_s

        "random_node_features": True,
        "n_random_node_features": 5,
        "relaxed": True,

        "T_max": 0.05,
        "N_warmup": 0,
        "N_anneal": 2000,
        "N_equil": 0,
        "stop_epochs": 800,

        ### TODO rework network and remove edge updates
        "n_hidden_neurons": 64,
        "n_features_list_prob": [64, 2],
        "n_features_list_nodes": [64, 64],
        "n_features_list_edges": [10],
        "n_features_list_messages": [64, 64],
        "n_features_list_encode": [30],
        "n_features_list_decode": [64],
        "n_message_passes": 2,
        "message_passing_weight_tied": False,
        "linear_message_passing": True,
        "edge_updates": False,
        "n_diffusion_steps": 1,
        "beta_factor": 0.1,
        "noise_potential": "annealed_obj",

        "time_conditioning": True,

        "project_name": args.project_name,
        "mean_aggr": False,
        "grad_clip": True,
        "messeage_concat": False,
        "graph_mode": "normal",
        "loss_alpha": 0.0,
        "MCMC_steps": 0,
        "train_mode": "REINFORCE",
        "inner_loop_steps": 2,
        "minib_diff_steps": 3,
        "minib_basis_states": 10,
        "graph_norm": False,
        "proj_method": "None",
        "diff_schedule": "DiffUCO",
        "mov_average": 0.05,
        "sampling_temp": 1.4,
        "n_sampling_rounds": 5,
        "n_test_basis_states": 20,
        "bfloat16": False,
        "T_target": 0.,
        "AnnealSchedule": "linear",
        "time_encoding": "one_hot",
        "lr_schedule": "cosine",
        "TD_k": 3,
        "clip_value": 0.2,
        "value_weighting": 0.65
    }

    if(overwrite):
        for key in flexible_config:
            if(key in config.keys()):
                config[key] = flexible_config[key]
            else:
                raise ValueError("key does not exist")

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_frac)
    if(args.deallocate):
        pass
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # from jax import config
    # config.update("jax_enable_x64", True)

    train = TrainMeanField(config)

    train.train()





if __name__ == "__main__":
    #run_PPO_experiment_func = lambda flex_conf: run_PPO_experiment_hydra()
    #run_PPO_experiment_func({"n":10})
    ### TODO test EngeryFunction Flag, for fully connected SK and for sparse mode
    ### TODO test T_min Flag
    ### TODO test global aggr feature Flag

    meanfield_run()
    #start_zero_T_run()
