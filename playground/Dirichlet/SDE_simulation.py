if(__name__ == "__main__"):
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    a = 2.0
    b = 2.
    s = 2/(a+b)
    X0 = 0.3
    T = 5.0  # Total time
    N = 1000  # Number of steps
    dt = T / N  # Time step

    # Initialize the array to store the simulated values
    X = np.zeros(N + 1)
    X[0] = X0

    # Generate the standard normal random variables
    epsilon = np.random.normal(0, 1, N)

    # Simulate the process
    for i in range(N):
        X[i + 1] = X[i] + s/2 * (a*(1 - X[i]) )* dt + np.sqrt(s*X[i] * (1 - X[i])) * np.sqrt(dt) * epsilon[i]

    # Plot the simulated process
    plt.plot(np.linspace(0, T, N + 1), X)
    plt.xlabel('Time')
    plt.ylabel('X_t')
    plt.title('Univariate Jacobi Diffusion Process')
    plt.show()