import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from eval_step_factor import plot_over_diff_eval_steps
from scipy.stats import wilcoxon, ttest_rel
import time
def read_and_get_mean_energy(wandb_run_id, eval_steps = 3, n_states = 8, mode = "test"):
    path = os.getcwd()
    path_to_models = os.path.dirname(os.path.dirname(path)) + "/Checkpoints"
    path_folder = f"{path_to_models}/{wandb_run_id}/"

    if not os.path.exists(path_folder):
        print(f"Folder not found: {path_folder}")
        raise ValueError("Folder not found")

    file_name = f"{wandb_run_id}_test_dict_eval_step_factor_{eval_steps}_{n_states}.pickle"

    # Read the pickle file
    with open(os.path.join(path_folder, file_name), 'rb') as f:
        test_dict = pickle.load(f)

    if('test/MaxCut_Value_CE' in test_dict.keys()):
        result_dict = {"mean_energy_CE": test_dict["test/MaxCut_Value_CE"],
                       "mean_energy": test_dict["test/MaxCut_Value"],
                       "overall_time": test_dict["test/overall_time"],
                       "forward_pass_time": test_dict["test/forward_pass_time"]
                       }
    else:
        result_dict = {"mean_energy_CE": test_dict["test/mean_energy_CE"],
                       "mean_energy": test_dict["test/mean_energy"],
                       "overall_time": test_dict["test/overall_time"],
                       "CE_time": test_dict["test/CE_time"],
                       "forward_pass_time": test_dict["test/forward_pass_time"]
                       }

    return result_dict

def read_and_get_best_energy(wandb_run_id, eval_steps = 3, n_states = 150, mode = "test"):
    path = os.getcwd()
    path_to_models = os.path.dirname(os.path.dirname(path)) + "/Checkpoints"
    path_folder = f"{path_to_models}/{wandb_run_id}/"

    if not os.path.exists(path_folder):
        print(f"Folder not found: {path_folder}")
        raise ValueError("Folder not found")

    file_name = f"{wandb_run_id}_test_dict_eval_step_factor_{eval_steps}_{n_states}.pickle"

    # Read the pickle file
    with open(os.path.join(path_folder, file_name), 'rb') as f:
        test_dict = pickle.load(f)

    if('test/best_MaxCut_Value_CE' in test_dict.keys()):
        result_dict = {"best_energy_CE": test_dict["test/best_MaxCut_Value_CE"],
                       "best_energy": test_dict["test/best_MaxCut_Value_CE"],
                       }
    else:
        result_dict = {"best_energy_CE": test_dict["test/mean_best_energy_CE"],
                       "best_energy": test_dict["test/mean_best_energy"],
                       "mean_gt_energy": test_dict['test/mean_gt_energy'],
                       }

    return result_dict

MaxCl_fKL = ["ud133dhi", "yp999f1m", "thpyvtjx"]
MaxCl_PPO = ["5uvh7t41", "l3ricjby", "flbniwwf"]
MaxCl_rKL = ["k1zc0zgg", "yxyr9urj", "l3s6eybg"]
MaxCL_small_dict = {
    "PPO": MaxCl_PPO,
    "rKL": MaxCl_rKL,
    "FKL": MaxCl_fKL,
}


MIS_small_fKL = ["otpu58r3", "9xoy68e6", "w3u4cer6"]
MIS_small_PPO = ["zk3wkaap", "91icd2vu", "fj1lym7o"]
MIS_small_rKL = ["m3h9mz5g", "olqaqfnl", "08i3m2dl"]
MIS_small_dict = {
    "PPO": MIS_small_PPO,
    "rKL": MIS_small_rKL,
    "fKL": MIS_small_fKL,

}

MIS_large_fKL = ["6rrd7m5b","rjcm5oto", "lgo8u2aq"]
MIS_large_PPO = ["rj161hwt", "eh5td2pi", "c8d4u3uo"]
MIS_large_rKL = ["cvv1wla0", "fuu10c4p", "00qoqw0s"]
MIS_large_dict = {
    "PPO": MIS_large_PPO,
    "rKL": MIS_large_rKL,
    "fKL": MIS_large_fKL,

}

MaxCut_large_fKL = ["r3ils8y0", "qidpkk4j", "96u1z4mu"]
MaxCut_large_PPO = ["oi3fyq7w", "4qmwye2w", "6irzwfyk"]
MaxCut_large_rKL = ["ubti92kx", "c11rjsun", "c6yoqwmp"]
MaxCut_large_dict = {
    "PPO": MaxCut_large_PPO,
    "rKL": MaxCut_large_rKL,
    "fKL": MaxCut_large_fKL,

}

MDS_large_fKL = ["1bxuoqyk", "7kitzopi", "uq21j7gp"]
MDS_large_PPO = ["x2z3k811", "ukunkwov", "t95aukxn"]
MDS_large_rKL = ["d0ipxvey", "oxsb0ors", "1fc4qeqq"]
MDS_large_dict = {
    "PPO": MDS_large_PPO,
    "rKL": MDS_large_rKL,
    "fKL": MDS_large_fKL,

}

##cosine
# MDS_large_fKL = ["uq21j7gp", "7kitzopi", "1bxuoqyk"]
# MDS_large_PPO = ["t95aukxn", "ukunkwov", "x2z3k811"]
# MDS_large_rKL = ["1fc4qeqq", "oxsb0ors", "d0ipxvey"]
# MDS_large_dict = {
#     "PPO": MDS_large_PPO,
#     "rKL": MDS_large_rKL,
#     "fKL": MDS_large_fKL,
# }

## one hot
MDS_large_fKL = ["x3mdgetb", "cpg13tch", "05juku5c"]
MDS_large_PPO = ["t95aukxn", "ukunkwov", "x2z3k811"] # cosine #["tmrwdhrm", "11ys3mm9", "t95aukxn"]
MDS_large_rKL = ["64dnrg5p", "107hsfqv", "0liz28ec"]
MDS_large_dict = {
    "PPO": MDS_large_PPO,
    "rKL": MDS_large_rKL,
    "fKL": MDS_large_fKL,

}

MaxCut_small_fKL = ["8ah3bsvm", "c1l3z0d4", "s2ug6f2y"]
MaxCut_small_PPO = ["9wstv9tl","m5l9z4j6", "nka5ez0d"]
MaxCut_small_rKL = ["114mqmhk", "t2ud5ttf", "icuxbpll"]
MaxCut_small_dict = {
    "PPO": MaxCut_small_PPO,
    "rKL": MaxCut_small_rKL,
    "fKL": MaxCut_small_fKL,

}

MDS_small_fKL = ["a7zogxmh", "df9rgk6a", "yjwopr68"]
MDS_small_PPO = ["qfrf58hx", "2ll1wcw1", "4rgz2hck"]
MDS_small_rKL = ["ydq3mn05", "xzc9mds3", "gk5s4nar"]
MDS_small_dict = {
    "PPO": MDS_small_PPO,
    "fKL": MDS_small_fKL,
    "rKL": MDS_small_rKL,

}

def seconds_to_hms(seconds):
    # Calculate hours, minutes, and remaining seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    # Return the formatted string as H:M:S
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def calc_mean_and_std(wandb_id_dict, ret_func):
    method_result_dict = {}
    for wandb_key in wandb_id_dict.keys():
        method_result_dict[wandb_key] = {}
        wanb_id_list = wandb_id_dict[wandb_key]
        results = {}

        for wandb_id in wanb_id_list:
            result_dict = ret_func(wandb_id)

            for key in result_dict.keys():
                if(key not in results.keys()):
                    results[key] = []
                    results[key].append(result_dict[key])
                else:
                    results[key].append(result_dict[key])

        for key in result_dict.keys():
            if("time" in key):
                method_result_dict[wandb_key][key] = results[key]
                #mean_time = time.strftime('%H:%M:%S', time.gmtime(np.mean(solutions_dict[key])))
                mean_time = seconds_to_hms(np.mean(results[key]))
                std_time = time.strftime('%H:%M:%S',
                                         time.gmtime(np.mean(np.std(results[key]) / np.sqrt(len(results[key])))))
                #print("time list", results[key])
                print(key, "$", mean_time, "\pm", std_time, "$")

            else:
                method_result_dict[wandb_key][key] = results[key]
                #print(wandb_key, key, "$", np.round(results[key], decimals=2))
                print(wandb_key, key , "$",np.round(np.mean(results[key]), decimals=2), "\pm ", np.round(np.std(results[key])/np.sqrt(len(results[key])), decimals = 2) , "$", len(results[key]))

    # for method_key in method_result_dict.keys():
    #     for method_key_2 in method_result_dict.keys():
    #
    #         if(method_key != method_key_2):
    #             wilcoxon_test(method_result_dict, method_key, method_key_2)
def wilcoxon_test(method_result_dict, method_1, method_2):

    for metric_key in method_result_dict[method_1].keys():
        metric_1 = np.array(method_result_dict[method_1][metric_key])
        metric_2 = np.array(method_result_dict[method_2][metric_key])
        diffs = metric_1 - metric_2
        print(metric_key, method_1, method_2)
        print(diffs)
        if(np.mean(abs(diffs)) != 0):
            print("ttest", ttest_rel(metric_1, metric_2, alternative = "less"))
            result = wilcoxon(diffs)
            print(metric_key, method_1, method_2, result, wilcoxon(diffs, alternative = "greater"), wilcoxon(diffs, alternative = "less"))


def best_energy_runs(GPU_num = "0", measure_time = False, n_samples = 150, batch_size = 1):
    ### TODO MIS large is missing here!
    wandb_dict_list = [MIS_large_dict]
    datasets = ["RB_iid_large"]

    method_run_list = []
    for method_dict, dataset in zip(wandb_dict_list, datasets):
        str = ""
        for key in method_dict:
            for method_wandb_id in method_dict[key]:
                str = str + " " + method_wandb_id

        overall_run_sting = f"python ConditionalExpectation.py  --GPU {GPU_num} --evaluation_factors 3 --n_samples {n_samples} --batch_size {batch_size} --dataset {dataset} --wandb_id {str};"
        method_run_list.append(overall_run_sting)

    print_str = ""
    for method in method_run_list:
        print_str += method + " "

    print(print_str)

def measure_time_runs(GPU_num = "0", measure_time = True, n_samples = 30, batch_size = 1):
    ### TODO MIS large is missing here!
    wandb_dict_list = [MIS_large_dict]
    datasets = ["RB_iid_large"]

    method_run_list = []
    for method_dict, dataset in zip(wandb_dict_list, datasets):
        str = ""
        for key in method_dict:
            for method_wandb_id in method_dict[key]:
                str = str + " " + method_wandb_id

        overall_run_sting = f"python ConditionalExpectation.py  --GPU {GPU_num} --evaluation_factors 3 --n_samples {n_samples} --batch_size {batch_size} --measure_time {measure_time} --dataset {dataset} --wandb_id {str};"
        method_run_list.append(overall_run_sting)

    print_str = ""
    for method in method_run_list:
        print_str += method + " "

    print(print_str)

def mean_energy_runs(GPU_num = "4", measure_time = False, n_samples = 30, batch_size = 30):
    wandb_dict_list = [  MaxCut_large_dict]
    datasets = [ "BA_large"]

    method_run_list = []
    for method_dict, dataset in zip(wandb_dict_list, datasets):
        str = ""
        for key in method_dict:
            for method_wandb_id in method_dict[key]:
                str = str + " " + method_wandb_id

        overall_run_sting = f"python ConditionalExpectation.py  --GPU {GPU_num} --evaluation_factors 3 --n_samples {n_samples} --batch_size {batch_size} --dataset {dataset} --wandb_id {str};"
        method_run_list.append(overall_run_sting)

    print_str = ""
    for method in method_run_list:
        print_str += method + " "

    print(print_str)

if(__name__ == "__main__"):
    ### TODO double check time for (r) results
    ### TODO run 150 states runs
    ### TODO run 30 states runs on MaxCut large, MDS large and MIS large
    print("MIS small")
    calc_mean_and_std(MIS_small_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MIS_small_dict, read_and_get_best_energy)
    print("MaxCL")
    calc_mean_and_std(MaxCL_small_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MaxCL_small_dict, read_and_get_best_energy)
    print("MIS large")
    calc_mean_and_std(MIS_large_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MIS_large_dict, read_and_get_best_energy)
    print("MaxCut large")
    calc_mean_and_std(MaxCut_large_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MaxCut_large_dict, read_and_get_best_energy)
    print("MaxCut small")
    calc_mean_and_std(MaxCut_small_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MaxCut_small_dict, read_and_get_best_energy)
    print("MDS small")
    calc_mean_and_std(MDS_small_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MDS_small_dict, read_and_get_best_energy)
    print("MDS large")
    calc_mean_and_std(MDS_large_dict, lambda a: read_and_get_mean_energy(a, eval_steps = 3, n_states = 30))
    calc_mean_and_std(MDS_large_dict, read_and_get_best_energy)
    #raise ValueError("")

    best_energy_runs(GPU_num="4")
    mean_energy_runs(GPU_num="4",batch_size = 20)
    measure_time_runs(GPU_num="4")

