from .AnnealedNoise import AnnealedNoiseDistr
from .BernoulliNoise import BernoulliNoiseDistr
from .CategoricalNoise import CategoricalNoseDistr
from .CombinedNoiseDistr import CombinedNoiseDistr

### TODO implement mixture of AnnealedNoise and Bernoulli Noise
noise_distribution_registry = {"annealed_obj": AnnealedNoiseDistr, "bernoulli": BernoulliNoiseDistr, "categorical": CategoricalNoseDistr, "combined": CombinedNoiseDistr}



def get_Noise_class(config):

    noise_distr_str = config["noise_potential"]

    if(noise_distr_str in noise_distribution_registry.keys()):
        noise_class = noise_distribution_registry[noise_distr_str]
    else:
        raise ValueError(f"Noise potential {noise_distr_str} is not implemented")

    return noise_class(config)