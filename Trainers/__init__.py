from .REINFORCE_Trainer import Reinforce
from .PPO_Trainer import PPO
from.ForwardKL_Trainer import ForwardKL

### TODO implement mixture of AnnealedNoise and Bernoulli Noise
Trainer_registry = {"REINFORCE": Reinforce, "PPO": PPO, "Forward_KL": ForwardKL}



def get_Trainer_class(config):
    train_mode_str = config["train_mode"]
    if(train_mode_str in Trainer_registry.keys()):
        noise_class = Trainer_registry[train_mode_str]
    else:
        raise ValueError(f"Train mode {train_mode_str} is not implemented")
    return noise_class