import sys
sys.path.append("..")
import os
import argparse

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--wandb_id', default = "",type = str, help='Switch ray into local mode for debugging')
parser.add_argument('--GPUs', type=int, nargs='+', default=[0, 1], help='List of GPU IDs')
parser.add_argument('--memory', default=0.92, type = float, help="GPU memory")


# Parse the arguments
args = parser.parse_args()


if(__name__ == "__main__"):
    # cd code/DiffUCO/playground/Clusters/Vega

    devices = args.GPUs

    GPU_string = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            GPU_string += str(devices[idx]) + ","
        else:
            GPU_string += str(devices[idx])

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
        f"#SBATCH -J {args.wandb_id}\n",
        f"#SBATCH -o {args.wandb_id}_logfile.out\n",
        f"#SBATCH -e {args.wandb_id}_logfile.err\n",
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
        f"python continue_training.py --wandb_id {args.wandb_id} --GPUs {' '.join(map(str, args.GPUs))} --memory {args.memory} \n"

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
