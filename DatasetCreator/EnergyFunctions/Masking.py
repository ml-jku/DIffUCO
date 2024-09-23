
import jax.numpy as jnp
import numpy as np


def _from_class_to_spins(self, sampled_class):
    bin_arr = np.unpackbits(sampled_class.reshape((-1, 1)).view(np.uint8), axis=1, count=self.n_sampled_sites, bitorder="little")
    spin_configuration = 2 * np.array(bin_arr, dtype=np.int32) - 1
    # print(f"num = {sampled_class}")
    # print(f"spins = {spin_configuration}")
    return spin_configuration

def make_NbNt_spins(n_sampled_sites, Nb):
    all_tokens = np.arange(0, 2 ** n_sampled_sites)
    all_spin_configurations = _from_class_to_spins(all_tokens)

    Nt_spins = np.zeros((all_tokens.shape[0], all_spin_configurations.shape[0]))
    Nt_spins[:, 0:n_sampled_sites] = all_spin_configurations
    Nt_spins = np.expand_dims(Nt_spins, axis=-1)

    Nb_Nt_spins = np.repeat(Nt_spins[np.newaxis, :, :, :], Nb, axis=0)
    NbNt_spins = np.reshape(Nb_Nt_spins, (Nb_Nt_spins.shape[0] * Nb_Nt_spins.shape[1], Nb_Nt_spins.shape[2], 1))
    return NbNt_spins


def mask( all_spin_configuration, NbNt_spins, Nb_sampled_spins):
    ### TODO move this elsewhere
    nbnt_all_spins = np.repeat(Nb_sampled_spins[:, np.newaxis, :self.env_step, :], all_tokens.shape[0], axis=1)
    nbnt_all_spins = np.reshape(nbnt_all_spins,(nbnt_all_spins.shape[0] * nbnt_all_spins.shape[1], nbnt_all_spins.shape[2], 1))
    NbNt_spins = np.concatenate([nbnt_all_spins, NbNt_spins], axis=1)
    violations = _compute_violations(NbNt_spins, self.init_EnergyJgraph)
    violations = 1 * (np.reshape(np.ravel(violations), (Nb_Nt_spins.shape[0], Nb_Nt_spins.shape[1])) > 0)
    mask = 1 - violations

    # print(violations)
    violations_per_state = jnp.sum(violations, axis=-1)
    if (np.any(violations_per_state >= 2 ** self.n_sampled_sites)):
        print("num_violations", violations_per_state)
        print("violations before", jnp.squeeze(self._compute_violations(self.Nb_spins, self.init_EnergyJgraph)))
        ValueError("There are too many violations")

    return mask

def _compute_violations(self, Nb_spins, H_graph):
    ### TODO use Hgraph that has no self loops here!

    if(self.EnergyFunction == "MIS" or "MaxCl" in self.EnergyFunction):
        Nb_bins = jnp.where(Nb_spins == 0, Nb_spins, (Nb_spins + 1) / 2)
        Nb_ext_fields_edges = 0.5*Nb_bins[:,H_graph.senders]*Nb_bins[:,H_graph.receivers]
    elif(self.EnergyFunction == "MVC"):
        Nb_bins = jnp.where(Nb_spins == 0, Nb_spins + 1, (Nb_spins + 1) / 2)
        Nb_ext_fields_edges = 0.5 * (1-Nb_bins[:, H_graph.senders]) * (1-Nb_bins[:, H_graph.receivers])
    Nb_energy_from_self_loops = jnp.sum(Nb_ext_fields_edges, axis = -2)

    return Nb_energy_from_self_loops