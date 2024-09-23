# Python script to generate a .txt file with the specified SLURM job script content
import os
def transfer_script(script_path, script_name):
    rsa_path = "/system/user/sanokows/.ssh/id_rsa"
    local_script_path_txt = script_path
    remote_script_path = "~/code/DiffUCO" + f"/argparse/scripts/{script_name}.txt"

    # local_CE = os.getcwd() + f"/ConditionalExpectation.py"
    # CE_file_path = "~/code/meanfield_annealing" + f"/ConditionalExpectation.py"
    # transfer_CE_command = f"scp -i {rsa_path} {local_CE} it4i-sanokow@karolina.it4i.cz:{CE_file_path}"
    ### TODO transfer script to server
    transfer_command = f"scp -i {rsa_path} {local_script_path_txt} it4i-sanokow@karolina.it4i.cz:{remote_script_path}"
    run_script = f'ssh -i {rsa_path} it4i-sanokow@karolina.it4i.cz "sbatch {remote_script_path}"'

    # os.system(f"mv {local_script_path_txt} {local_script_path_sh}")
    os.system(f"dos2unix {local_script_path_txt}")
    os.system(f"cat {local_script_path_txt}")
    os.system(transfer_command)
    # os.system(transfer_CE_command)
    os.system(run_script)


def create_slurm_script(config):

    CO_problem = config['CO_problem']
    dataset = config['dataset']
    N_anneal = config['N_anneal']
    lr = config['lr']
    n_GNN_layers = config['n_GNN_layers']
    temp = config['temp']
    noise_potential = config['noise_potential']
    n_diffusion_steps = config['n_diffusion_steps']
    batch_size = config['batch_size']
    n_basis_states = config['n_basis_states']
    train_mode = config['train_mode']
    minib_diff_steps = config['minib_diff_steps']
    n_rand_nodes = config['n_rand_nodes']
    diff_schedule =config['diff_schedule']
    proj_name = f"Dataset_fixed_{train_mode}_{n_diffusion_steps}_{train_mode}"

    seed = config['seed']
    nGPUs = config['nGPUs']


    GPU_string = ""
    for i in range(nGPUs):
        GPU_string += str(i) + ' '

    lines = [
        "#!/bin/bash\n",
        "\n",
        "#SBATCH --account DD-23-97\n",
        "#SBATCH --partition qgpu\n",
        "#SBATCH --time 2-00:00:00\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gpus-per-node {nGPUs}\n",
        "#SBATCH --tasks-per-node=1\n",
        f"#SBATCH -J {dataset}_{CO_problem}\n",
        f"#SBATCH -o {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.out\n",
        f"#SBATCH -e {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.err\n",
        "#SBATCH --mail-type=BEGIN,ABORT,END\n",
        "#SBATCH --mail-user=sebastian.sanokowski@jku.at\n",
        "\n",
        "source activate rayjay_clone\n",
        "cd ~/code/DiffUCO/\n",
        "conda activate rayjay_clone\n",
        "\n",
        "nvidia-smi\n",
        "\n",
        "# run script\n",
        f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} --n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} --minib_diff_steps {minib_diff_steps}  --stop_epochs 1500 --mem_frac 0.98\n"
    ]


    import uuid
    filename = f"{uuid.uuid4()}.txt"
    with open(filename, 'w') as file:
        file.writelines(lines)

    print("file with filname", filename ,"was created")

    abs_path = os.path.abspath(filename)
    directory = os.path.dirname(abs_path)
    print(f"File created at: {abs_path}")
    print(f"Directory: {directory}")
    return abs_path, filename

def create_and_transfer(config):
    script_path, script_name = create_slurm_script(config)
    transfer_script(script_path, script_name)

train_modes = ["REINFORCE"]
lrs = [0.001]
temps = [0.15, 0.05]
n_diffusion_step_list = [12]
noise_distributions = ["bernoulli"] #["combined", "annealed_obj", "bernoulli"]
batch_size = 70
base_diff_steps = 4
for train_mode in train_modes:
    for lr in lrs:
        for temp in temps:
            for n_diffusion_steps in n_diffusion_step_list:
                for noise_distribution in noise_distributions:
                    if train_mode == "REINFORCE":
                        config = {
                            "CO_problem": "MIS",
                            "dataset": "RB_iid_100",
                            "N_anneal": 2000,
                            "lr": lr,
                            'n_GNN_layers': 8,
                            "temp":  temp,
                            'noise_potential': noise_distribution,
                            'n_diffusion_steps': n_diffusion_steps,
                            'batch_size': int(base_diff_steps*batch_size/n_diffusion_steps),
                            'n_basis_states': 10,
                            'train_mode': train_mode,
                            'minib_diff_steps': base_diff_steps,
                            'n_rand_nodes': 3,
                            'diff_schedule': "exp",
                            'seed': 123,
                            'nGPUs': 1,
                        }
                        create_and_transfer(config)
                    else:
                        config = {
                            "CO_problem": "MIS",
                            "dataset": "RB_iid_100",
                            "N_anneal": 2000,
                            "lr": lr,
                            'n_GNN_layers': 8,
                            "temp":  temp,
                            'noise_potential': noise_distribution,
                            'n_diffusion_steps': n_diffusion_steps,
                            'batch_size': batch_size,#int(3*batch_size/n_diffusion_steps),
                            'n_basis_states': 10,
                            'train_mode': train_mode,
                            'minib_diff_steps': base_diff_steps,
                            'n_rand_nodes': 3,
                            'diff_schedule': "exp",
                            'seed': 123,
                            'nGPUs': 1,
                        }
                        create_and_transfer(config)
