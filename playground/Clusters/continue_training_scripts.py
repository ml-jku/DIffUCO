import sys
sys.path.append("..")
from create_and_run_scripts import create_and_transfer_continue_script

if(__name__ == "__main__"):
    ### TODO use whole node by running several jobs per node
    ### This is meluxina?
    wandb_ids = ["huuxp6dj", "pdz1tnjk", "703l8dui", "06uyle4b"]
    GPU_list = [1, 1, 1, 1]

    if(len(wandb_ids) != len(GPU_list)):
        raise ValueError("lenghts must be the same!")

    create_and_transfer_continue_script(wandb_ids, GPU_list)