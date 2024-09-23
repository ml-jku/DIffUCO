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

    print("average Energy is", MCMC_chain[idx])



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

    stop_criterion(acf_values, MCMC_chain)
    # Check convergence
    if np.all(np.abs(acf_values[1:])/(len(acf_values)-1) < threshold):
        print("The chain seems to have converged based on the autocorrelation function.")
    else:
        print("The chain may not have fully converged yet.")

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

# Example usage
if __name__ == "__main__":
    # Simulated MCMC chain (for demonstration purposes)
    import pickle
    import os

    path = os.getcwd()
    path_to_models = os.path.dirname(os.path.dirname(path)) + "/Checkpoints"

    def load_MCMC_chain( run_id, stuff_name="unbiased_sampling_log_dict"):
        path_folder = f"{path_to_models}/{run_id}/"

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        file_name = f"{run_id}_stuff_dict_{stuff_name}.pickle"

        with open(os.path.join(path_folder, file_name), 'rb') as f:
            log_dict = pickle.load(f)
        return log_dict


    run_id = "f0cszhfv"
    log_dict = load_MCMC_chain(run_id)
    MCMC_chain = log_dict["figures"]["internal_energies_MCMC"]["y_axis"]

    unbiased_X_seq = log_dict["samples"]["unbiased_X_sequences"][0]
    biased_X_seq = log_dict["samples"]["biased_X_sequences"][0]
    plot_X_sequences(biased_X_seq, title="biased")
    #plot_X_sequences(unbiased_X_seq, title = "unbiased")

    # Compute and plot autocorrelation
    acf_values = compute_acf(MCMC_chain, nlags=len(MCMC_chain))
    plot_acf(acf_values, MCMC_chain, threshold=0.05)