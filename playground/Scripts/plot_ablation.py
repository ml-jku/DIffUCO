if(__name__ == "__main__"):
    from matplotlib import pyplot as plt
    n_diff_steps = [3, 6,9, 12]

    Forward_KL = [0.05, 0.0389, 0.0362, 0.0368] #### TODO update
    Reverse_KL = [0.031, 0.02096, 0.0157, 0.0139]
    Reverse_KL_RL = [0.029, 0.01512, 0.0139, 0.0132]

    plt.figure()
    plt.plot(n_diff_steps, Forward_KL, label = "Forward KL")
    plt.plot(n_diff_steps, Reverse_KL, label =  "Reverse KL")
    plt.plot(n_diff_steps, Reverse_KL_RL, label = r"Reverse KL RL")
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(n_diff_steps, Forward_KL, label = "Forward KL")
    plt.plot(n_diff_steps, Reverse_KL, label =  "Reverse KL")
    plt.plot(n_diff_steps, Reverse_KL_RL, label = r"Reverse KL RL")
    plt.legend()
    plt.show()
    pass