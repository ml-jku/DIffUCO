import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RB_iid_large', help='Define the dataset')
parser.add_argument('--problem_type', default='MIS', help='Define the dataset')

args = parser.parse_args()


if(__name__ == "__main__"):
    rsa_path = "/system/user/sanokows/.ssh/id_rsa"
    ### TODO karolina must load from different location
    ### also transfer code here?
    ### add script

    dataset_types = ["no_norm"]
    for dataset_type in dataset_types:
        dataset = args.dataset
        datasplits = ["train", "val"]
        for datasplit in datasplits:
            dataset_path = f"/system/user/publicwork/sanokows/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/{dataset_type}/{dataset}/{datasplit}/"
            remote_dataset_path = f"/mnt/proj2/dd-23-97/{dataset_type}/{dataset}/{datasplit}/"

            remove_command = f'ssh -i {rsa_path} it4i-sanokow@karolina.it4i.cz "rm -rf {remote_dataset_path}"'
            create_folder_command = f'ssh -i {rsa_path} it4i-sanokow@karolina.it4i.cz "mkdir -p {remote_dataset_path}"'
            transfer_command = f"scp -r -i {rsa_path} {dataset_path} it4i-sanokow@karolina.it4i.cz:{remote_dataset_path}"

            os.system(remove_command)
            os.system(create_folder_command)
            os.system(transfer_command)

