import sys
sys.path.append("..")
import os
import argparse

parser = argparse.ArgumentParser(description='Process some arguments.')

# Add arguments to the parser
parser.add_argument('--lrs', type=float, default=0.001, help='Learning rate')
parser.add_argument('--relaxed', action='store_true', help='Use relaxed mode')
parser.add_argument('--GPUs', type=int, nargs='+', default=[0, 1], help='List of GPU IDs')
parser.add_argument('--n_GNN_layers', type=int, default=6, help='Number of GNN layers')
parser.add_argument('--temps', type=float, default=0.1, help='Temperature')
parser.add_argument('--IsingMode', type=str, default='BA_large', help='Ising Mode')
parser.add_argument('--EnergyFunction', type=str, default='MDS', help='Energy Function')
parser.add_argument('--mode', type=str, default='Diffusion', help='Mode')
parser.add_argument('--N_anneal', type=int, default=2000, help='Number of annealing steps')
parser.add_argument('--beta_factor', type=float, default=1.0, help='Beta factor')
parser.add_argument('--n_diffusion_steps', type=int, default=6, help='Number of diffusion steps')
parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
parser.add_argument('--n_basis_states', type=int, default=8, help='Number of basis states')
parser.add_argument('--noise_potential', type=str, default='bernoulli', help='Noise potential')
parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
parser.add_argument('--project_name', type=str, default='final_runs', help='Project name')
parser.add_argument('--n_rand_nodes', type=int, default=3, help='Number of random nodes')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--graph_mode', type=str, default='normal', help='Graph mode')
parser.add_argument('--train_mode', type=str, default='REINFORCE', help='Training mode')
parser.add_argument('--inner_loop_steps', type=int, default=1, help='Number of inner loop steps')
parser.add_argument('--diff_schedule', type=str, default='exp', help='Diffusion schedule')
parser.add_argument('--minib_diff_steps', type=int, default=6, help='Mini-batch diffusion steps')
parser.add_argument('--minib_basis_states', type=int, default=8, help='Mini-batch basis states')
parser.add_argument('--stop_epochs', type=int, default=1500, help='Number of stop epochs')
parser.add_argument('--mem_frac', type=float, default=0.92, help='Memory fraction')
parser.add_argument('--mov_average', type=float, default=0.099, help='mov_average')
parser.add_argument('--clip_value', type=float, default=0.2, help='clip_value')

# Parse the arguments
args = parser.parse_args()

# Create the argument string
arg_str = (f"--lrs {args.lrs} "
           f"{'--relaxed' if args.relaxed else ''} "
           f"--GPUs {' '.join(map(str, args.GPUs))} "
           f"--n_GNN_layers {args.n_GNN_layers} "
           f"--temps {args.temps} "
           f"--IsingMode {args.IsingMode} "
           f"--EnergyFunction {args.EnergyFunction} "
           f"--mode {args.mode} "
           f"--N_anneal {args.N_anneal} "
           f"--beta_factor {args.beta_factor} "
           f"--n_diffusion_steps {args.n_diffusion_steps} "
           f"--batch_size {args.batch_size} "
           f"--n_basis_states {args.n_basis_states} "
           f"--noise_potential {args.noise_potential} "
           f"{'--multi_gpu' if args.multi_gpu else ''} "
           f"--project_name {args.project_name} "
           f"--n_rand_nodes {args.n_rand_nodes} "
           f"--seed {args.seed} "
           f"--graph_mode {args.graph_mode} "
           f"--train_mode {args.train_mode} "
           f"--inner_loop_steps {args.inner_loop_steps} "
           f"--diff_schedule {args.diff_schedule} "
           f"--minib_diff_steps {args.minib_diff_steps} "
           f"--minib_basis_states {args.minib_basis_states} "
           f"--stop_epochs {args.stop_epochs} "
           f"--mem_frac {args.mem_frac} "
           f"--mov_average {args.mov_average} "
           f"--clip_value {args.clip_value} ")

# Print the argument string
print(arg_str)

if(__name__ == "__main__"):
    # cd code/DiffUCO/playground/Clusters/Vega

    devices = args.GPUs

    GPU_string = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            GPU_string += str(devices[idx]) + ","
        else:
            GPU_string += str(devices[idx])
    seed = args.seed
    user_path = "/ceph/hpc/home/eusebastians"
    init_lines = [
        "#!/bin/bash\n",
        "#SBATCH --partition=gpu"
        "\n",
        # f"#SBATCH --qos={qos}\n"
        "#SBATCH --time=2-00:00:00\n",
        "#SBATCH --mem=16G\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gres=gpu:{len(args.GPUs)}\n",
        "#SBATCH --tasks-per-node=1\n",
        "#SBATCH --cpus-per-task=4\n",
        f"#SBATCH -J {args.IsingMode}_{args.EnergyFunction}_{seed}\n",
        f"#SBATCH -o {args.IsingMode}_{args.EnergyFunction}_{args.train_mode}_{args.n_diffusion_steps}_{seed}_logfile.out\n",
        f"#SBATCH -e {args.IsingMode}_{args.EnergyFunction}_{args.train_mode}_{args.n_diffusion_steps}_{seed}_logfile.err\n",
        "#SBATCH --mail-type=BEGIN,ABORT,END\n",
        "#SBATCH --mail-user=sebastian.sanokowski@jku.at\n",
        "\n",
        f"cd {user_path}/code/DiffUCO\n",
        f"pwd\n",
        "module load Anaconda3\n",
        "source activate rayjay_clone\n"
        "conda activate rayjay_clone\n",
        "\n",
        "nvidia-smi\n",
    ]


    overall_lines = []
    overall_lines.extend(init_lines)

    script_lines = [
        f"python argparse_ray_main.py {arg_str} \n"

    ]

    overall_lines.extend(script_lines)

    import uuid
    filename = f"{uuid.uuid4()}.txt"
    with open(filename, 'w') as file:
        file.writelines(overall_lines)

    print("file with filname", filename, "was created")

    abs_path = os.path.abspath(filename)
    directory = os.path.dirname(abs_path)
    print(f"File created at: {abs_path}")
    print(f"Directory: {directory}")

    run_script = f"sbatch {abs_path}"

    os.system(run_script)
