import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


def stop_criterion(acf_values, MCMC_chain, C = 5):
    lags = np.arange(len(acf_values))
    int_tau =  1 + 2*np.cumsum(acf_values[1:])
    M = lags[1:] / C

    idx = None
    for i in range(M.shape[0]):
        if M[i] > int_tau[i]:
            idx = i

    print("average MCMC Energy is", MCMC_chain[idx])
    return MCMC_chain[idx]



def compute_acf(chain, nlags=40):
    """
    Compute the autocorrelation function for the given MCMC chain.

    Parameters:
    chain (array-like): The MCMC chain.
    nlags (int): Number of lags to compute the autocorrelation for.

    Returns:
    acf_values (array): Autocorrelation values for the lags.
    """
    acf_values = acf(chain, nlags=nlags, fft=True)
    return acf_values

def plot_acf(acf_values, chain, threshold=0.05):
    """
    Plot the autocorrelation function and assess convergence.

    Parameters:
    acf_values (array): The autocorrelation values.
    threshold (float): The threshold below which we consider the chain to have converged.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(chain, color='blue', alpha=0.7)
    plt.title('MCMC Chain')
    plt.xlabel('Iteration')
    plt.ylabel('Value')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 2)
    lags = np.arange(len(acf_values))
    plt.plot(lags, acf_values, color='blue', alpha=0.7)
    plt.axhline(y=0.05, color='red', linestyle='--', label=f'Threshold = 0.05')
    plt.axhline(y=-0.05, color='red', linestyle='--')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')

    C = 5
    plt.subplot(3, 1, 3)
    lags = np.arange(len(acf_values))
    plt.plot(lags[1:], 1 + 2*np.cumsum(acf_values[1:]), color='blue', alpha=0.7)
    plt.plot(lags[1:], lags[1:]/C, color='blue', alpha=0.7)
    plt.xlabel('Lag')
    plt.ylabel('cumsum Autocorrelation')

    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()
    # Check convergence
    if np.all(np.abs(acf_values[1:])/(len(acf_values)-1) < threshold):
        print("The chain seems to have converged based on the autocorrelation function.")
    else:
        print("The chain may not have fully converged yet.")
    return stop_criterion(acf_values, chain)

def plot_X_sequences(X_seq,n_trials = 5,n = 5, title = ""):

    X_sample = np.array(X_seq,dtype=np.int16)
    N = int(np.sqrt(X_sample.shape[1]-1))
    for nn in range(n_trials):
        plt.figure()
        plt.title(title)
        for i in range(X_sample.shape[0]):
            plt.subplot(int(X_sample.shape[0]/n+1), n, i +1 )
            X = X_sample[i,0:N*N,nn]
            plt.imshow(np.reshape(X, (N,N)))
        plt.show()

    pass


def plot_result_dict(run_id):
    print("Results of", run_id)
    results_dict = load_MCMC_chain(run_id, stuff_name="unbiased_sampling_results_dict")

    sampling_temps_scaled_list = list(results_dict.keys())
    seeds = len(list(results_dict[0.0].keys()))

    fig = plt.figure()
    for sampling_temp in sampling_temps_scaled_list:
        free_energies = np.mean(np.array(
            [results_dict[sampling_temp][seed]["free_energies"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                axis=0)
        free_energies_std = np.std(np.array(
            [results_dict[sampling_temp][seed]["free_energies"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                   axis=0) / np.sqrt(seeds)
        n_states = results_dict[sampling_temp][0]["free_energies"]["x_axis"]
        plt.title(
            f"Free Energies by number of samples \nSampling Temp: {sampling_temp} \n")
        plt.errorbar(n_states, free_energies, yerr=free_energies_std, fmt="-x", alpha=0.5,
                     label=f"sampling_temp = {sampling_temp}")
        #plt.axhline(y=gt_free_energy, color='r', linestyle='-')
        print("free energy at", sampling_temp, "is $",  free_energies[-1] , "\pm"  , free_energies_std[-1], "$")
        plt.legend()
        plt.ylabel("Free Energies")
        plt.xlabel("Number of Samples")
        plt.tight_layout()
    plt.close("all")

    for sampling_temp in sampling_temps_scaled_list:
        entropies = np.mean(np.array(
            [results_dict[sampling_temp][seed]["entropies"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                axis=0)
        entropies_std = np.std(np.array(
            [results_dict[sampling_temp][seed]["entropies"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                   axis=0) / np.sqrt(seeds)
        n_states = results_dict[sampling_temp][0]["entropies"]["x_axis"]
        plt.title(
            f"Free Energies by number of samples \nSampling Temp: {sampling_temp} \n")
        plt.errorbar(n_states, entropies, yerr=entropies_std, fmt="-x", alpha=0.5,
                     label=f"sampling_temp = {sampling_temp}")
        #plt.axhline(y=gt_free_energy, color='r', linestyle='-')
        print("entropies at", sampling_temp, "is $",  entropies[-1] , "\pm"  , entropies_std[-1], "$")
        plt.legend()
        plt.ylabel("entropies")
        plt.xlabel("Number of Samples")
        plt.tight_layout()
    plt.close("all")

    fig = plt.figure()
    for sampling_temp in sampling_temps_scaled_list:
        free_energies = np.mean(np.array(
            [results_dict[sampling_temp][seed]["internal_energies"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                axis=0)
        free_energies_std = np.std(np.array(
            [results_dict[sampling_temp][seed]["internal_energies"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                   axis=0) / np.sqrt(seeds)
        n_states = results_dict[sampling_temp][0]["internal_energies"]["x_axis"]
        plt.title(
            f"internal energies by number of samples \nSampling Temp: {sampling_temp} \n")
        plt.errorbar(n_states, free_energies, yerr=free_energies_std, fmt="-x", alpha=0.5,
                     label=f"sampling_temp = {sampling_temp}")
        #plt.axhline(y=gt_internal_energy, color='r', linestyle='-')
        print("internal energies at", sampling_temp, "is $",  free_energies[-1] , "\pm", free_energies_std[-1], "$")
        plt.legend()
        plt.ylabel("internal energies")
        plt.xlabel("Number of Samples")
        plt.tight_layout()
    plt.close("all")

    fig = plt.figure()
    for sampling_temp in sampling_temps_scaled_list:
        free_energies = np.mean(np.array(
            [results_dict[sampling_temp][seed]["internal_energies_MCMC"]["y_axis"] for seed in
             results_dict[sampling_temp]]), axis=0)
        free_energies_std = np.std(np.array(
            [results_dict[sampling_temp][seed]["internal_energies_MCMC"]["y_axis"] for seed in
             results_dict[sampling_temp]]), axis=0) / np.sqrt(seeds)
        n_states = results_dict[sampling_temp][0]["internal_energies_MCMC"]["x_axis"]
        print("here", n_states.shape, free_energies.shape)
        plt.title(
            f"internal energies MCMC by number of samples \nSampling Temp: {sampling_temp} \n")
        plt.errorbar(n_states, free_energies, yerr=free_energies_std, fmt="-x", alpha=0.5,
                     label=f"sampling_temp = {sampling_temp}")
        #plt.axhline(y=gt_internal_energy, color='r', linestyle='-')
        print("internal energy MCMC at", sampling_temp, "is $",  free_energies[-1] , "\pm", free_energies_std[-1], "$")
        plt.legend()
        plt.ylabel("internal energies MCMC")
        plt.xlabel("Number of Samples")
        plt.tight_layout()
    plt.close("all")

    fig = plt.figure()
    for sampling_temp in sampling_temps_scaled_list:
        free_emergies_abs_error = np.mean(np.array(
            [results_dict[sampling_temp][seed]["free_energies_err"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                          axis=0)
        free_emergies_abs_error_std = np.std(np.array(
            [results_dict[sampling_temp][seed]["free_energies_err"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                             axis=0) / np.sqrt(seeds)
        n_states = results_dict[sampling_temp][0]["free_energies_err"]["x_axis"]
        plt.title(
            f"Absolute Error by number of samples \nSampling Temp: {sampling_temp} \n")
        plt.errorbar(n_states, free_emergies_abs_error, yerr=free_emergies_abs_error_std, fmt="-x", alpha=0.5,
                     label=f"sampling_temp = {sampling_temp}")
        print("free energy error at", sampling_temp, "is $",  free_emergies_abs_error[-1], "\pm", free_emergies_abs_error_std[-1] , "$")
    plt.legend()
    plt.ylabel("Absolute Error")
    plt.yscale("log")
    plt.xlabel("Number of Samples")
    plt.tight_layout()
    plt.close("all")

    fig = plt.figure()
    for sampling_temp in sampling_temps_scaled_list:
        effective_sample_size = np.mean(np.array(
            [results_dict[sampling_temp][seed]["eff_sample_size"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                        axis=0)
        effective_sample_size_std = np.std(np.array(
            [results_dict[sampling_temp][seed]["eff_sample_size"]["y_axis"] for seed in results_dict[sampling_temp]]),
                                           axis=0) / np.sqrt(seeds)
        n_states = results_dict[sampling_temp][0]["eff_sample_size"]["x_axis"]
        plt.title(
            f"Absolute Error by number of samples \nSampling Temp: {sampling_temp} \n")
        plt.errorbar(n_states, effective_sample_size / n_states, yerr=effective_sample_size_std / n_states, fmt="-x",
                     alpha=0.5, label=f"sampling_temp = {sampling_temp}")

        print("eff sample size at", sampling_temp, "is $",  effective_sample_size[-1] / n_states[-1], "\pm", effective_sample_size_std[-1] / n_states[-1], "$")
        print("len seeds", seeds)
    plt.plot(n_states, 1 / n_states, "-", label="worst eff sample size")
    plt.legend()
    plt.yscale("log")
    plt.ylabel("effective_sample_size")
    plt.xlabel("Number of Samples")
    plt.tight_layout()
    plt.close("all")

# Example usage
if __name__ == "__main__":
    # Simulated MCMC chain (for demonstration purposes)
    import pickle
    import os

    current_file_path = os.path.abspath(__file__)
    # Get the parent directory of the current file
    parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    path_to_models = parent_folder + "/Checkpoints"

    def load_MCMC_chain( run_id, stuff_name="unbiased_sampling_log_dict"):
        path_folder = f"{path_to_models}/{run_id}/"

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        file_name = f"{run_id}_stuff_dict_{stuff_name}.pickle"

        with open(os.path.join(path_folder, file_name), 'rb') as f:
            log_dict = pickle.load(f)
        MCMC_log_dict = log_dict[0.0]
        MCMC_energies = []
        for seed in MCMC_log_dict.keys():
            MCMC_chain = MCMC_log_dict[seed]["internal_energies_MCMC"]["y_axis"]
            acf_values = compute_acf(MCMC_chain, nlags=len(MCMC_chain))
            MCMC_energy = plot_acf(acf_values, MCMC_chain, threshold=0.05)
            MCMC_energies.append(MCMC_energy)

        print("final MCMC energy", np.mean(MCMC_energies), np.std(MCMC_energies)/np.sqrt(len(MCMC_energies)))
        return log_dict

    #ISING?
    
    ## FKL
    # run_id = "qkfzunur"
    # plot_result_dict(run_id)

    #PPO
    # run_id = "ewmsen06"
    # plot_result_dict(run_id)

    # # PPO
    # run_id = "sw5qr5e6"
    # plot_result_dict(run_id)

    ### SPIN GLASSES ?
    # REINFORCE
    run_id = "m7fi604s"
    plot_result_dict(run_id)

    # PPO
    run_id = "5j75a3k9"
    plot_result_dict(run_id)

    # fKL
    run_id = "4hl3jr35"
    plot_result_dict(run_id)
