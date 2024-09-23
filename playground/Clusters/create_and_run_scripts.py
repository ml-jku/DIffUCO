import os
from Meluxina import user_path, user_id, data_path, server_str, project_id
def create_and_transfer(config, proj_name = None):

    script_path, script_name = create_meluxina_script(config, proj_name = proj_name)

    transfer_script(script_path, script_name)

def create_and_transfer_continue_script(wandb_ids, nGPU_list):
    script_path, script_name = continue_Meluxina_script(wandb_ids, nGPU_list)
    transfer_script(script_path, script_name)

def transfer_script(script_path, script_name):
    rsa_path = "/system/user/sanokows/.ssh/meluxina"
    local_script_path_txt = script_path
    remote_script_path = "~/code/DiffUCO" + f"/argparse/scripts/{script_name}.txt"

    # local_CE = os.getcwd() + f"/ConditionalExpectation.py"
    # CE_file_path = "~/code/meanfield_annealing" + f"/ConditionalExpectation.py"
    # transfer_CE_command = f"scp -i {rsa_path} {local_CE} it4i-sanokow@karolina.it4i.cz:{CE_file_path}"
    ### TODO transfer script to server
    transfer_command = f"scp -P 8822 -i {rsa_path} {local_script_path_txt} {server_str}:{remote_script_path}"
    run_script = f'ssh -i {rsa_path} {server_str} -p 8822 "sbatch {remote_script_path}"'

    # os.system(f"mv {local_script_path_txt} {local_script_path_sh}")
    os.system(f"dos2unix {local_script_path_txt}")
    os.system(f"cat {local_script_path_txt}")
    os.system(transfer_command)
    # os.system(transfer_CE_command)
    os.system(run_script)
def create_slurm_script(config, proj_name = None):
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
    mem_frac =config['mem_frac']
    if(proj_name == None):
        proj_name = f"Dataset_fixed_{train_mode}_{n_diffusion_steps}_{train_mode}_{batch_size}"

    seed = config['seed']
    nGPUs = config['nGPUs']
    qos = config['qos']


    GPU_string = ""
    for i in range(nGPUs):
        GPU_string += str(i) + ' '

    lines = [
        "#!/bin/bash\n",
        "#SBATCH --partition=gpu"
        "\n",
        f"#SBATCH --account={project_id}\n",
        f"#SBATCH --qos={qos}\n"
        "#SBATCH --time=2-00:00:00\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gpus={nGPUs}\n",
        f"#SBATCH --gpus-per-node={nGPUs}\n",
        "#SBATCH --tasks-per-node=1\n",
        f"#SBATCH -J {dataset}_{CO_problem}\n",
        f"#SBATCH -o {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.out\n",
        f"#SBATCH -e {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.err\n",
        "#SBATCH --mail-type=BEGIN,ABORT,END\n",
        "#SBATCH --mail-user=sebastian.sanokowski@jku.at\n",
        "\n",
        f"cd {user_path}/code/DiffUCO\n",
        f"pwd\n",
        "source activate rayjay_clone\n"
        "conda activate rayjay_clone\n",
        "\n",
        "nvidia-smi\n",
        "\n",
        "# run script\n",
        f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} "
        f"--n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} "
        f"--minib_diff_steps {minib_diff_steps} --minib_basis_states {n_basis_states}  --stop_epochs 1500 --mem_frac {mem_frac} &\n"

        f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} "
        f"--n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} "
        f"--minib_diff_steps {minib_diff_steps} --minib_basis_states {n_basis_states}  --stop_epochs 1500 --mem_frac {mem_frac} &\n"

        "wait\n"

        'echo "Both scripts have completed."\n'
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

def create_slurm_script_Ising(config, proj_name = None):
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
    mem_frac =config['mem_frac']

    T_target =  config['T_target']
    n_sampling_rounds =  config['n_sampling_rounds']
    n_test_basis_states =  config['n_test_basis_states']
    minib_basis_states =  config['minib_basis_states']
    n_hidden_neurons =  config['n_hidden_neurons']

    if(proj_name == None):
        proj_name = f"Dataset_fixed_{train_mode}_{n_diffusion_steps}_{train_mode}_{batch_size}"

    seed = config['seed']
    nGPUs = config['nGPUs']
    qos = config['qos']


    GPU_string = ""
    for i in range(nGPUs):
        GPU_string += str(i) + ' '

    lines = [
        "#!/bin/bash\n",
        "#SBATCH --partition=gpu"
        "\n",
        f"#SBATCH --account={project_id}\n",
        f"#SBATCH --qos={qos}\n"
        "#SBATCH --time=2-00:00:00\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gpus={nGPUs}\n",
        f"#SBATCH --gpus-per-node={nGPUs}\n",
        "#SBATCH --tasks-per-node=1\n",
        f"#SBATCH -J {dataset}_{CO_problem}\n",
        f"#SBATCH -o {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.out\n",
        f"#SBATCH -e {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.err\n",
        "#SBATCH --mail-type=BEGIN,ABORT,END\n",
        "#SBATCH --mail-user=sebastian.sanokowski@jku.at\n",
        "\n",
        f"cd {user_path}/code/DiffUCO\n",
        f"pwd\n",
        "source activate rayjay_clone\n"
        "conda activate rayjay_clone\n",
        "\n",
        "nvidia-smi\n",
        "\n",
        "# run script\n",
        f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} "
        f"--n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} "
        f"--minib_diff_steps {minib_diff_steps} --minib_basis_states {n_basis_states}  --stop_epochs 1500 --mem_frac {mem_frac} --T_target {T_target} --n_sampling_rounds {n_sampling_rounds} --n_test_basis_states {n_test_basis_states} --minib_basis_states {minib_basis_states} --n_hidden_neurons {n_hidden_neurons} --time_encoding cosine &\n"

        f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} "
        f"--n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} "
        f"--minib_diff_steps {minib_diff_steps} --minib_basis_states {n_basis_states}  --stop_epochs 1500 --mem_frac {mem_frac} --T_target {T_target} --n_sampling_rounds {n_sampling_rounds} --n_test_basis_states {n_test_basis_states} --minib_basis_states {minib_basis_states} --n_hidden_neurons {n_hidden_neurons} --time_encoding cosine&\n"

        "wait\n"

        'echo "Both scripts have completed."\n'
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

def create_meluxina_script(configs, proj_name = None):
    config = configs[0]
    CO_problem = config['CO_problem']
    dataset = config['dataset']
    n_diffusion_steps = config['n_diffusion_steps']
    batch_size = config['batch_size']
    train_mode = config['train_mode']
    mov_average = config['mov_average']

    if(proj_name == None):
        proj_name = f"Dataset_fixed_{train_mode}_{n_diffusion_steps}_{train_mode}_{batch_size}"

    seed = config['seed']
    nGPUs = config['nGPUs']
    qos = config['qos']


    GPU_string = ""
    for i in range(nGPUs):
        GPU_string += str(i) + ' '

    init_lines = [
        "#!/bin/bash\n",
        "#SBATCH --partition=gpu"
        "\n",
        f"#SBATCH --account={project_id}\n",
        f"#SBATCH --qos={qos}\n"
        "#SBATCH --time=2-00:00:00\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gpus={4}\n",
        f"#SBATCH --gpus-per-node={4}\n",
        f"#SBATCH --tasks-per-node={len(configs)}\n",
        f"#SBATCH -J {dataset}_{CO_problem}\n",
        f"#SBATCH -o {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.out\n",
        f"#SBATCH -e {dataset}_{CO_problem}_{train_mode}_{n_diffusion_steps}_logfile.err\n",
        "#SBATCH --mail-type=BEGIN,ABORT,END\n",
        "#SBATCH --mail-user=sebastian.sanokowski@jku.at\n",
        "\n",
        f"cd {user_path}/code/DiffUCO\n",
        f"pwd\n",
        "source activate rayjay_clone\n"
        "conda activate rayjay_clone\n",
        "\n",
        "nvidia-smi\n",
    ]


    end_script_lines = [
        "wait\n"

        'echo "Both scripts have completed."\n']

    overall_lines = []
    overall_lines.extend(init_lines)

    used_gpus = 0
    for config in configs:
        overall_lines.extend(create_script_lines(config, used_gpus, proj_name= proj_name))
        used_gpus += config['nGPUs']

    overall_lines.extend(end_script_lines)

    import uuid
    filename = f"{uuid.uuid4()}.txt"
    with open(filename, 'w') as file:
        file.writelines(overall_lines)

    print("file with filname", filename ,"was created")

    abs_path = os.path.abspath(filename)
    directory = os.path.dirname(abs_path)
    print(f"File created at: {abs_path}")
    print(f"Directory: {directory}")
    return abs_path, filename


def create_script_lines(config, used_gpus, proj_name = None ):
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
    mov_average = config["mov_average"]
    mem_frac =config['mem_frac']
    if(proj_name == None):
        proj_name = f"Dataset_fixed_{train_mode}_{n_diffusion_steps}_{train_mode}_{batch_size}"

    seed = config['seed']
    nGPUs = config['nGPUs']


    GPU_string = ""
    for i in range(nGPUs):
        GPU_string += str(i + used_gpus) + ' '

    if(CO_problem == "IsingModel"):
        T_target = config['T_target']
        n_sampling_rounds = config['n_sampling_rounds']
        n_test_basis_states = config['n_test_basis_states']
        minib_basis_states = config['minib_basis_states']
        n_hidden_neurons = config['n_hidden_neurons']

        script_lines = [f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} "
        f"--n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} --mov_average {mov_average}"
        f"--minib_diff_steps {minib_diff_steps} --minib_basis_states {n_basis_states}  --stop_epochs 10000 --mem_frac {mem_frac} --T_target {T_target} --n_sampling_rounds {n_sampling_rounds} --n_test_basis_states {n_test_basis_states} --minib_basis_states {minib_basis_states} --n_hidden_neurons {n_hidden_neurons} --time_encoding cosine &\n"]
    else:
        script_lines = [
            f"python argparse_ray_main.py --lrs {lr} --relaxed --GPUs {GPU_string} --n_GNN_layers {n_GNN_layers} --temps {temp} --IsingMode {dataset} --EnergyFunction {CO_problem} --mode Diffusion --N_anneal {N_anneal} --beta_factor 1. --n_diffusion_steps {n_diffusion_steps} --batch_size {batch_size} --n_basis_states {n_basis_states} --noise_potential {noise_potential} --multi_gpu --project_name {proj_name} "
            f"--n_rand_nodes {n_rand_nodes} --seed {seed} --graph_mode normal --train_mode {train_mode} --inner_loop_steps 1 --diff_schedule {diff_schedule} --mov_average {mov_average} "
            f"--minib_diff_steps {minib_diff_steps} --minib_basis_states {n_basis_states}  --stop_epochs 1500 --mem_frac {mem_frac} &\n"

        ]
    return script_lines

def create_continue_script_lines(wandb_id, nGPUs, used_gpus):

    GPU_string = ""
    for i in range(nGPUs):
        GPU_string += str(i + used_gpus) + ' '

    script_lines = [
        f"python continue_training.py --wandb_id {wandb_id} --GPUs {GPU_string}  &\n"

    ]
    return script_lines

def continue_Meluxina_script(wandb_ids, nGPU_list):


    init_lines = [
        "#!/bin/bash\n",
        "#SBATCH --partition=gpu"
        "\n",
        f"#SBATCH --account={project_id}\n",
        f"#SBATCH --qos=default\n"
        "#SBATCH --time=2-00:00:00\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gpus={4}\n",
        f"#SBATCH --gpus-per-node={4}\n",
        "#SBATCH --tasks-per-node=1\n",
        f"#SBATCH -J continue_training\n",
        f"#SBATCH -o continue_training_logfile.out\n",
        f"#SBATCH -e continue_training_logfile.err\n",
        "#SBATCH --mail-type=BEGIN,ABORT,END\n",
        "#SBATCH --mail-user=sebastian.sanokowski@jku.at\n",
        "\n",
        f"cd {user_path}/code/DiffUCO\n",
        f"pwd\n",
        "source activate rayjay_clone\n"
        "conda activate rayjay_clone\n",
        "\n",
        "nvidia-smi\n",
    ]


    end_script_lines = [
        "wait\n"

        'echo "Both scripts have completed."\n']

    overall_lines = []
    overall_lines.extend(init_lines)

    used_gpus = 0
    for wandb_id, nGPUs in zip(wandb_ids, nGPU_list):
        overall_lines.extend(create_continue_script_lines(wandb_id, nGPUs, used_gpus))
        used_gpus += nGPUs

    overall_lines.extend(end_script_lines)

    import uuid
    filename = f"{uuid.uuid4()}.txt"
    with open(filename, 'w') as file:
        file.writelines(overall_lines)

    print("file with filname", filename ,"was created")

    abs_path = os.path.abspath(filename)
    directory = os.path.dirname(abs_path)
    print(f"File created at: {abs_path}")
    print(f"Directory: {directory}")
    return abs_path, filename