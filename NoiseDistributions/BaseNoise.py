from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax

class BaseNoiseDistr(ABC):

    def __init__(self, config):
        self.config = config
        self.beta_factor = self.config["beta_factor"]
        self.n_diffusion_steps =  self.config["n_diffusion_steps"]
        self.beta_list = []
        for i in range(self.n_diffusion_steps):
            self.beta_list.append(self.beta_t_func(i, self.n_diffusion_steps, self.beta_factor))

        self.beta_arr = jnp.flip(jnp.array(self.beta_list), axis=-1)

        print("noise potential", self.config["noise_potential"])
        print("beta is")
        print(self.beta_arr)
        print("______________")

    @abstractmethod
    def get_log_p_T_0(self):
        pass

    @abstractmethod
    def calc_noise_loss(self):
        pass

    @abstractmethod
    def beta_t_func(self):
        pass

    @abstractmethod
    def combine_losses(self):
        pass

    @abstractmethod
    def calculate_noise_distr_reward(self):
        pass


    def calc_noise_step_relaxed(self):
        raise ValueError("calc_noise_step_relaxed function not implemented")
        pass

    @abstractmethod
    def calc_noise_step(self):
        pass

    def beta_t_func(self, t, n_diffusion_steps, k = 0.):
        if(self.config["diff_schedule"] == "own"):
            tau = 30
            beta = 1 / (tau*((n_diffusion_steps-t-1)/n_diffusion_steps) + 2)
        elif(self.config["diff_schedule"] == "exp"):
            tau = 6
            x = (n_diffusion_steps-t-1)/n_diffusion_steps
            beta = 2**(-tau*x)*0.5

        elif (self.config["diff_schedule"] == "DiffUCO"):
            beta = 1 / (n_diffusion_steps - t + 1)
        elif (self.config["diff_schedule"] == "Ho"):
            beta = (1-(n_diffusion_steps - (t+1))/(n_diffusion_steps -t))/2
            beta = max([beta, 0.01])
        elif (self.config["diff_schedule"] == "Campell"):
            b = 100
            a = 0.01
            Tau =n_diffusion_steps

            beta = 0.5*(1-np.exp(Tau*a*(b**(t/Tau) - b**((t+1)/Tau))))
            beta = max([beta, 0.01])
            alpha_T = np.exp(Tau * a * (1 - b))
            print("alpha hat T = ", alpha_T)

        else:
            beta = 1 / (n_diffusion_steps - t + 2)
        return beta

    def get_gamma_t(self, t_idx):

        return self.beta_arr[t_idx]



