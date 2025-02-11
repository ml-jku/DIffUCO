import os
import argparse
from train import TrainMeanField


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_ids', default = ["7knl4x32"],type = str, help='wandb ids', nargs = "+")
parser.add_argument('--GPUs', default=["0"], type = str, help='Define Nb', nargs = "+")
parser.add_argument('--memory', default=0.92, type = float, help="GPU memory")
parser.add_argument('--expl', default=[0.], type = float, help="amount of exploration", nargs = "+")
parser.add_argument('--seeds', default=1, type = int, help="num_seeds")
parser.add_argument('--n_sampling_rounds', default=1000, type = int, help="Number of sampling rounds")
parser.add_argument('--sampling_modes', default=["temps"], type = str, help="eps or temps", nargs = "+")
parser.add_argument('--n_test_basis_states', default=400, type = int, help="eps or temps")
args = parser.parse_args()
# python evaluate.py --wandb_id hlmw6stl  --GPU MIG-c69ed117-8436-51d1-b4db-183ea0228cd6 --exp 0. 0.01 0.05 --n_sampling_rounds 400 --sampling_modes temps

def SpinGlass_16x16():
    ### TODO test influce of eps
    ### PPO id # 9c8t3sl0
    #python evaluate.py --wandb_id sw5qr5e6  --GPU 2 --exp 0. --n_sampling_rounds 400 --sampling_modes temps --n_test_basis_states 1200 --seeds 9
    ###fKL # ygwjc1f4
    #python evaluate.py --wandb_id 4hl3jr35  --GPU 3 --exp 0. --n_sampling_rounds 400 --sampling_modes temps --n_test_basis_states 1200 --seeds 9
    ### rKL # wr7i4oy9
    #python evaluate.py --wandb_id m7fi604s  --GPU 6 --exp 0. --n_sampling_rounds 400 --sampling_modes temps --n_test_basis_states 1200 --seeds 9   
    pass

def good_wandb_ids():
    fKL = {"f0cszhfv": {"n_diff_steps": 30, "T_start": 10} , "1838jcin": {"n_diff_steps": 60, "T_start": 10}, "hlmw6stl": {"n_diff_steps": 100, "T_start": 10}}
    PPO = {"9lu3ahbm": {"n_diff_steps": 30, "T_start": 10}, "ts7dch2k": {"n_diff_steps": 30, "T_start": 4}}
    REINFORCE = {"da75rnt": {"n_diff_steps": 10, "T_start": 10}}


    # python evaluate.py --wandb_id bp0pthmf  --GPU 7 --exp 0. --n_sampling_rounds 400 --sampling_modes temps
    # python evaluate.py --wandb_id qkfzunur  --GPU 4 --exp 0. --n_sampling_rounds 400 --n_test_basis_states 1200 --seeds 3  --sampling_modes temps


def _8x8_ablation():
    # python evaluate.py --wandb_id 23n6g8f2 usjpghdv sywdxxhb  --GPU MIG-c6766c68-2ea4-5e48-b9d4-f0d93f1beeed --exp 0. --n_sampling_rounds 1000 --seeds 3 --sampling_modes temps

    pass

def evaluate():
    devices = args.GPUs

    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str)

    if (len(args.GPUs) > 1):
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


    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory)
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    config = {"jit": True}

    for wandb_id in args.wandb_ids:
        train = TrainMeanField(config, load_wandb_id=wandb_id)
        train.wandb_old_run_id = wandb_id
        for sampling_mode in args.sampling_modes:
            train.test_ubiased_estimator(args.expl, args.seeds, args.n_sampling_rounds, sampling_mode, args.n_test_basis_states)

if(__name__ == "__main__"):
    evaluate()