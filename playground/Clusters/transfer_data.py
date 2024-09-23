import os
import argparse
from Meluxina import user_path, user_id, data_path, server_str

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BA_small', help='Define the dataset')
parser.add_argument('--problem_type', default='MDS', help='Define the dataset')

args = parser.parse_args()


if(__name__ == "__main__"):
    rsa_path = "/system/user/sanokows/.ssh/meluxina"
    ### TODO karolina must load from different location
    ### also transfer code here?
    ### add script

    dataset_types = ["no_norm"]
    for dataset_type in dataset_types:
        dataset = args.dataset
        datasplits = ["train", "val"]
        for datasplit in datasplits:
            dataset_path = f"/system/user/publicwork/sanokows/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/{dataset_type}/{dataset}/{datasplit}/"
            remote_dataset_path = f"{data_path}/{dataset_type}/{dataset}/{datasplit}/"

            remove_command = f'ssh -i {rsa_path} {server_str} -p 8822 "rm -rf {remote_dataset_path}"'
            create_folder_command = f'ssh -i {rsa_path} {server_str} -p 8822 "mkdir -p {remote_dataset_path}"'
            transfer_command = f"scp -P 8822 -r -i {rsa_path} {dataset_path} {server_str}:{remote_dataset_path}"

            os.system(remove_command)
            os.system(create_folder_command)
            os.system(transfer_command)