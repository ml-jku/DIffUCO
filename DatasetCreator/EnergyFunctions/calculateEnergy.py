from . import MaxCut, MIS

def calc(H_graph,spins,EnergyFunction):
    if (EnergyFunction == "MaxCut" or EnergyFunction == "WeightedMaxCut"):
        Energy = MaxCut.calcEnergy(H_graph, spins)
    elif (EnergyFunction == "MIS"):
        Energy = MIS.calcEnergy(H_graph, spins)
    else:
        ValueError("Energy Function is not Valid")
    return Energy

def calc_sparse(H_graph,spins,EnergyFunction):
    if (EnergyFunction == "MaxCut" or EnergyFunction == "WeightedMaxCut"):
        Energy = MaxCut.calcEnergy_sparse(H_graph, spins)
    elif (EnergyFunction == "MIS"):
        Energy = MIS.calcEnergy_sparse(H_graph, spins)
    else:
        ValueError("Energy Function is not Valid")
    return Energy