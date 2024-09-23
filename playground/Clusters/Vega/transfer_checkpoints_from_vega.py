import os

def load_data():

    Checkpoint_path = "~/code/DiffUCO/Checkpoints"
    Upload_folder_path = "/system/user/publicwork/sanokows/DiffUCO"

    #load_command = f"scp -r -i {rsa_path} it4i-sanokow@karolina.it4i.cz:{Checkpoint_path} {Download_folder_path}"
    load_command = f"scp -r {Checkpoint_path} gorilla:{Upload_folder_path}"
    print(load_command)
    #os.system(load_command)

    ### TODO make similar script that starts runs on the server
    ### 1. transfer .pbs script -> run .pbs script


if(__name__ == "__main__"):
    load_data()