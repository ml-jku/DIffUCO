import os
from Meluxina import user_path, user_id, data_path, server_str

def load_data():
    rsa_path = "/system/user/sanokows/.ssh/meluxina"

    Checkpoint_path = "~/code/DiffUCO/Checkpoints"
    Download_folder_path = "/system/user/publicwork/sanokows/DiffUCO"

    #load_command = f"scp -r -i {rsa_path} it4i-sanokow@karolina.it4i.cz:{Checkpoint_path} {Download_folder_path}"
    load_command = f"scp -P 8822 -r -i {rsa_path} {server_str}:{Checkpoint_path}  {Download_folder_path}"
    os.system(load_command)

    ### TODO make similar script that starts runs on the server
    ### 1. transfer .pbs script -> run .pbs script


if(__name__ == "__main__"):
    load_data()