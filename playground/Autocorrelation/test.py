import numpy as np
import matplotlib.pyplot as plt


# Function to generate a sinusoidal signal
def generate_signal(frequency, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

# Function to compute the autocorrelation of a signal
def compute_autocorrelation(signal):
    autocorrelation = np.convolve(signal, signal, mode='full')
    # Normalize the autocorrelation
    autocorrelation = autocorrelation[autocorrelation.size // 2 + 1:]
    autocorrelation = autocorrelation / autocorrelation[0]
    return autocorrelation

if(__name__ == "__main__"):
    # Parameters for the signal
    frequency = 5  # Frequency of the sinusoidal signal
    sampling_rate = 100  # Sampling rate
    integration_time = 30
    duration = 2  # Duration in seconds

    # Generate the signal
    t, signal = generate_signal(frequency, sampling_rate, duration)

    # Compute the autocorrelation
    autocorrelation = compute_autocorrelation(signal)

    # Time axis for autocorrelation
    lags = np.arange(0, autocorrelation.shape[0])

    integrated_autocorrelation_time = 1 + 2*np.sum(autocorrelation[:integration_time])

    # Plot the signal
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Generated Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the autocorrelation function
    plt.subplot(2, 1, 2)
    plt.plot(lags, autocorrelation)
    plt.title('Autocorrelation Function' + str(np.round(integrated_autocorrelation_time, 4)))
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')

    plt.tight_layout()
    plt.show()