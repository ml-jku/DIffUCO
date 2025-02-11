def convert_seconds(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{days}d:{hours}h:{minutes}m:{seconds}s"

# Example usage

dict_key = {
    "fKL": [27,52,84,101],
    "REINFROCE": [23, 40,61,81],
    "PPO": [29, 56,80,107]
}

for key in dict_key:
    second_list = dict_key[key]
    for second in second_list:
        seconds = second*2000
        formatted_time = convert_seconds(seconds)
        print(key, formatted_time)  # Output: 11d:10h:44m:14s


import numpy as np
sol_dict_SpinGlass = {"fKL": [0.11737, 0.11469, 0.10511, 0.078384, 0.078384, 0.078384, 0.078384, 0.078384, 0.066126, 0.0277724], 
                      "PPO": [0.015, 0.000818, 0.000818, 0.000794, 0.00079463, 0.00079461, 0.00079456, 7.7669*10**-8, 6.8778*10**-8, 6.6624*10**(-8)]}

L = 10
for key in sol_dict_SpinGlass:
    res_list = sol_dict_SpinGlass[key]
    print("key", "mean", np.mean(res_list)/L**2,"std", np.std(res_list)/L**2)