import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NxNLattice_8x8', help='Define the dataset')
parser.add_argument('--problem_type', default='IsingModel', help='Define the dataset')

args = parser.parse_args()


if(__name__ == "__main__"):
    # cd code/DiffUCO/playground/Clusters/Vega
    #python load_from_remote_to_vega.py
    loc_path = os.path.abspath('../../..')
    dataset_types = ["no_norm"]
    for dataset_type in dataset_types:
        dataset = args.dataset
        datasplits = ["train", "val"]
        for datasplit in datasplits:
            dataset_path = f"/system/user/publicwork/sanokows/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/{dataset_type}/{dataset}/{datasplit}/"
            remote_dataset_path = f"{loc_path}/DatasetCreator/loadGraphDatasets/DatasetSolutions/{dataset_type}/{dataset}/{datasplit}/"


            remove_command = f"rm -rf {remote_dataset_path}"
            create_folder_command = f"mkdir -p {remote_dataset_path}"
            transfer_command = f"rsync -avz gorilla:{dataset_path} {remote_dataset_path}"

            os.system(remove_command)
            os.system(create_folder_command)
            os.system(transfer_command)