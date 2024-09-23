import pickle
from matplotlib import pyplot as plt
import numpy as np
if(__name__ == "__main__"):
    ms = 10
    lw = 2
    seed_124 = [1.0034913397127125, 1.0031790229488986, 1.0024732691911076, 1.0028479083050519, 1.0015318213053757, 1.0012958657168334, 1.0019762840781166, 1.0008380153437673, 1.000228016979988, 1.0]
    seed_123 = [1.0039706626531428, 1.0034328446775336, 1.0033345543274088, 1.0029048038293333, 1.0019460021888877,1.0020887873470283, 1.0018211826193626, 1.0008536761797675, 1.0006769039129433, 1.0]

    import itertools
    marker = itertools.cycle(('-d', '-P', '-v', '-o', '-*', "-<", "->"))
    gmarker = itertools.cycle(('--v', '--o', '--^'))

    with open("RB_200_MVC_results.pickle", "rb") as f:
        res_dict = pickle.load(f)

    print(res_dict.keys())

    p_values = res_dict["p_values"]
    plt.figure()

    if(True):
        for key in res_dict["Gurobi_results"]:
            mean_Gurobi_AR = res_dict["Gurobi_results"][key]["mean"]
            std_Gurobi_AR = res_dict["Gurobi_results"][key]["std"]
            plt.errorbar(p_values, mean_Gurobi_AR, yerr=std_Gurobi_AR, fmt=next(gmarker), label=key, markersize = ms, linewidth=lw, color = "gray")

    plt.errorbar(p_values, np.array(res_dict["DB-Greedy"]["mean"]), yerr=res_dict["DB-Greedy"]["std"], fmt=next(marker), label=r"DB-Greedy $t = $" + "{:.2f}".format(0.01), markersize = ms, linewidth=lw)

    plt.errorbar(p_values, np.array(res_dict["VAG-CO"]["mean"]) , yerr=np.array(res_dict["VAG-CO"]["std"]), fmt = next(marker), label = r"VAG-CO $t = $" + "{:.2f}".format(0.20), markersize = ms, linewidth=lw)
    #plt.errorbar(MFA_no_anneal_ps, 1 + MFA_no_anneal_dict["results"]["mean"], yerr=MFA_no_anneal_dict["results"]["std"], fmt = next(marker), label = rf"{MFA_no_anneal_dict['method']}: CE $t = $" + "{:.2f}".format(0.1), markersize = ms, linewidth=lw)
    plt.errorbar(p_values, np.array(res_dict["EGN"]["no_anneal"]["mean"]), yerr=np.array(res_dict["EGN"]["no_anneal"]["std"])-1, fmt = next(marker), label = rf"EGN: CE $t = $" + "{:.2f}".format(0.1), markersize = ms, linewidth=lw)
    plt.errorbar(p_values, np.array(res_dict["EGN"]["anneal"]["mean"]), yerr=np.array(res_dict["EGN"]["anneal"]["std"])-1, fmt = next(marker), label = rf"EGN-anneal: CE $t = $" + "{:.2f}".format(0.1), markersize = ms, linewidth=lw)
    #plt.errorbar(MFA_no_anneal_ps, 1 + MFAnneal_dict["results"]["mean"], yerr=MFAnneal_dict["results"]["std"], fmt = next(marker), label = rf"{MFAnneal_dict['method']}: CE $t = $" + "{:.2f}".format(0.1), markersize = ms, linewidth=lw)
    plt.errorbar(p_values, np.array(res_dict["DiffUCO"]["mean"]), yerr=np.array(res_dict["DiffUCO"]["std"]), fmt=next(marker), label=rf"DiffUCO: CE-ST$_8: \, \, t = $" + f"{0.02}", markersize=ms,linewidth=lw)
    #plt.errorbar(EGN_ann_ps, 1 + EGN_ann_rel_error, yerr=EGN_ann_std_error, fmt = next(marker), label = r"EGN-Anneal: CE $t = $" + "{:.2f}".format(0.1), markersize = ms, linewidth=lw)
    #plt.errorbar(EGN_ps, 1 + EGN_rel_error, yerr=EGN_std_error, fmt = next(marker), label = r"EGN: CE $t = $" + "{:.2f}".format(0.1), markersize = ms, linewidth=lw)
    #plt.errorbar(EGN_ps, AR_masking, yerr=0*EGN_std_error, fmt = next(marker), label = r"VAG-CO masking: CE $t = $" + "XX", markersize = ms, linewidth=lw)
    # for key in AR_res_dict:
    #     AR_ps = AR_res_dict[key]["p_list"]
    #     model_AR_list = AR_res_dict[key]["AR_list"]
    #     std_AR_list = AR_res_dict[key]["std_AR_list"]
    #     plt.errorbar(AR_ps, model_AR_list , yerr=std_AR_list, fmt = "x", label = f"AR + Anneal checkpoint best mean; Nb = {key}")
    plt.plot(p_values, seed_123, next(marker), label=rf"DiffUCO: RL$: \, \, t = $" + f"XX", markersize=ms,linewidth=lw)


    plt.grid()
    plt.legend(fontsize = 11.5, loc = "upper center", ncol = 2)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.ylim(top = 1.035)
    plt.xlabel(r"$p$", fontsize=30)
    plt.ylabel(r"$AR^*$", fontsize=24)
    plt.tight_layout()
    plt.savefig("MVC_RB_200_Rebuttal.png", dpi=1200)
    plt.show()