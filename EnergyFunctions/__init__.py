from .MISEnergy import MISEnergyClass
from .MVCEnergy import MVCEnergyClass
from .MDSEnergy import MDSEnergyClass
from .MaxClEnergy import MaxClEnergyClass
from .MaxCutEnergy import MaxCutEnergyClass
from .TSPEnergy import TSPEnergyClass
from .IsingModelEnergy import IsingModelEnergyClass
from .SpinGlass import SpinGlassEnergyClass

noise_distribution_registry = {"MIS": MISEnergyClass, "MVC": MVCEnergyClass, "MaxCl": MaxClEnergyClass, "SpinGlass": SpinGlassEnergyClass, "SpinGlassUniform": SpinGlassEnergyClass,
                               "TSP": TSPEnergyClass, "MaxCut": MaxCutEnergyClass, "MDS": MDSEnergyClass, "IsingModel": IsingModelEnergyClass}



def get_Energy_class(config):

    noise_distr_str = config["problem_name"]

    if(noise_distr_str in noise_distribution_registry.keys()):
        Energy_class = noise_distribution_registry[noise_distr_str]
    else:
        raise ValueError(f"CO Problem {noise_distr_str} is not implemented")

    return Energy_class(config)