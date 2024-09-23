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


def create_slurm_script(wandb_id):

    lines = [
        "#!/bin/bash\n",
        "\n",
        "#SBATCH --account DD-23-97\n",
        "#SBATCH --partition qgpu\n",
        "#SBATCH --time 2-00:00:00\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --gpus-per-node {1}\n",
        "#SBATCH --tasks-per-node=1\n",
        f"#SBATCH -J {wandb_id}\n",
        f"#SBATCH -o {wandb_id}_logfile.out\n",
        f"#SBATCH -e {wandb_id}_logfile.err\n",
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
        f"python continue_training.py --wandb_id {wandb_id} --GPUs 0 --memory 0.97"
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


wandb_ids = ["g0wrg6mn"]

for wandb_id in wandb_ids:
    create_and_transfer(wandb_id)
