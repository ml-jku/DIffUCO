import numpy as np


def get_split_datasets(Dataset, H_batch_size):
    arr = []
    for idx, (data,_) in enumerate(Dataset):
        num_nodes = data.nodes.shape[0]
        arr.append([idx, num_nodes])

    arr = np.array(arr)

    equl_sized_arrays = split_array(arr, H_batch_size)

    sorted_idxs = [ np.array(list)[:,0] for list in equl_sized_arrays]
    return sorted_idxs


def split_array(arr, n_splits, seed = 0):
    np.random.seed(seed)
    sorted_arr = sorted(zip(arr[:,0], arr[:,1]),key = lambda x: x[1])
    sorted_arr = np.array(sorted_arr)

    to_split_list = []
    for i in range(sorted_arr.shape[0]):
        if(i %2 == 0):
            arr = sorted_arr[sorted_arr.shape[0] - i - 1]
            to_split_list.append(arr)
        else:
            arr = sorted_arr[i]
            to_split_list.append(arr)

    print(to_split_list)
    to_split_list = np.array(to_split_list)
    to_split_list = np.array_split(to_split_list, n_splits)

    return to_split_list