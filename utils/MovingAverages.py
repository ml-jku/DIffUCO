import numpy as np
import jax.numpy as jnp
from functools import partial
import jax

class MovingAverage:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        self.step_count = 0
        self.mean_value = jnp.array([0.])
        self.std_value = jnp.array([0.])

    def update_mov_averages(self, data):
        if(self.alpha == -1):
            return 0., 1.
        else:
            self.mean_value = self.mean_step(self.mean_value, data)
            self.std_value = self.std_step(self.std_value, data)
            self.step_count += 1
        return self.mean_value, self.std_value

    def update_rule(self, value, data, factor):
        if(self.step_count == 0):
            value = data
        else:
            value = factor * data + (1-factor)*value

        return value

    @partial(jax.jit, static_argnums=(0,))
    def mean_step(self, mean_value, data):
        mean_data = jnp.mean(data)
        return self.update_rule(mean_value, mean_data, self.alpha)

    @partial(jax.jit, static_argnums=(0,))
    def std_step(self, std_value, data):
        std_data = jnp.std(data)
        return self.update_rule(std_value, std_data, self.beta)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_average(self, rewards, mov_average_reward, mov_std_reward):
        normed_rewards = (rewards - mov_average_reward)/(mov_std_reward + 10**-10)
        return normed_rewards





if(__name__ == "__main__"):
    from matplotlib import pyplot as plt

    for b in [False, True]:
        jax.config.update('jax_disable_jit', b)
        alphas = [0.2]
        for alpha in alphas:
            T = 1.
            N_anneal = 200
            N_warmup = 10
            epochs = N_anneal + N_warmup
            MovAvg = MovingAverage(alpha, alpha)

            T_curr_list = []
            mov_data = []
            buffer = []
            epoch_arr = np.arange(0, epochs)
            for epoch in epoch_arr:
                print(epoch, alpha)
                T_curr = jnp.sin(10*2*jnp.pi*epoch/ epochs) + 10*epoch/epochs
                T_curr_list.append(T_curr)
                buffer.append(T_curr)

                mean, std = MovAvg.update_mov_averages(T_curr)

                mov_data.append([epoch, mean, std])
                # if(len(buffer) > 0.001*epochs):
                #     mean, std = MovAvg.update_mov_averages(np.array(buffer))
                #     mov_data.append([epoch, mean, std])
                #     buffer = []

            plt.figure()
            plt.title(str(alpha))
            plt.xlabel(r"$N_\mathrm{epoch}$", fontsize = 22)
            plt.ylabel("Temperature", fontsize = 15)
            #plt.ylim(bottom = 0., top = 1.05)
            plt.plot(np.arange(0, epochs), T_curr_list)
            plt.plot([data[0] for data in mov_data], [data[1] for data in mov_data], label = "mov average")
            plt.plot([data[0] for data in mov_data], [data[2] for data in mov_data], label = "std average")
            plt.axvline(x=N_warmup, c='red', linestyle='-.', linewidth=2.)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.show()