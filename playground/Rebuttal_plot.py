from matplotlib import pyplot as plt
import numpy as np

def innit_fontsize():
    import matplotlib.pyplot as plt
    SMALL_SIZE = 8
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if(__name__ == "__main__"):
    innit_fontsize()
    ks = [1,2,4,6,8,10, 12]
    times = [52, 31, 19, 15, 13, 14, 22]
    rel_error = [[0.02771, 0.02335, 0.02567], [0.02761, 0.02335, 0.02567], [0.02771, 0.02335, 0.02567],
                 [0.02762, 0.02344, 0.02558], [0.02762, 0.02335, 0.02567], [0.02771, 0.02324, 0.02567], [0.02762, 0.02335, 0.02567]]

    markers = iter(["x", "v", "<", ">"])
    linestyles = iter(["-", "-", "--", "--"])
    colors = iter(["red", "blue", "green", "m"])

    plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.errorbar(ks, [np.mean(el)for el in rel_error], yerr=[np.std(el)/np.sqrt(len(el))for el in rel_error],
                 label=r"$\epsilon_{rel}$", fmt = next(markers),  color = next(colors), linestyle = next(linestyles) , markersize = 10, mew=4)
    ax1.set_xlabel(r"ST parameter $k$",  fontsize = 19)
    ax1.set_ylabel(r"$\epsilon_{rel}$", color='black',  fontsize = 24)
    ax1.set_ylim(top = 0.03, bottom = 0.02)

    ax2.plot(ks, times, label = r"evaluation time", marker = next(markers), color = next(colors), linestyle = next(linestyles), markersize = 10, mew=4)
    ax2.set_ylabel('seconds', color="black",  fontsize = 19)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), ncol = 1, fontsize = 20)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 1.), ncol = 1, fontsize = 16)
    #ax2.grid()
    #plt.legend()
    plt.tight_layout()
    plt.savefig("CE-ST_ablation.png", dpi=1200)
    plt.show()

    ### calc n_query
    CO_problems = ["MaxCut_small", "MaxCut_large", "MIS_small", "MIS_large", "MDS_small", "MDS_large", "MaxCl_small"]
    training_times = ["00:08:01:00", "00:16:21:00", "00:13:34:00", "5:13:35:00", "00:18:13:00", "00:12:45:00", "2:51:17:00" ]
    time_per_graph_DiffCO = [13/500, 53/500, 13/500, 65/500, 13/500, 40/500, 13/500]
    time_per_graph_Gurobi = [60, 300, (47*60+34)/500, (1*60*60+ 24*60+12)/500, (1*60 + 47)/500, (12*60 + 48)/500, (60*1 + 55)/500]

    import datetime
    for idx, training_time in enumerate(training_times):
        d, h, m, s = training_time.split(':')
        #print(int(datetime.timedelta(days = int(d), hours=int(h), minutes=int(m), seconds=int(s)).total_seconds()), "s")
        training_time = int(datetime.timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s)).total_seconds())

        gurobi_time = time_per_graph_Gurobi[idx]
        model_time = time_per_graph_DiffCO[idx]

        n_query = training_time/(gurobi_time - model_time)

        CO_problem = CO_problems[idx]
        print(CO_problem)
        print(int(n_query)+1)

    import wandb


    def load_logs(project_name, run_id):
        # Initialize W&B run for the desired project and run ID
        api = wandb.Api()
        run = api.run(f"{project_name}/{run_id}")

        # Fetch logged data
        logged_data = run.history()

        return logged_data, run

    def add_to_dict(run, key_name = "train/losses/log_p_0_T"):

        measure = []
        for idx, chunk in enumerate(run.scan_history()):
            # Print the logged data for the current chunk
            data = [chunk[key_name] for point in chunk if key_name in point]

            val_rel_error = chunk[key_name]

            if (val_rel_error != None):
                measure.append(val_rel_error)
        return measure

    project_name = "diff_steps_Diffusion_RB_iid_100_MIS_relaxed_True_deeper"
    run_id = "ui5npvol"

    # Load logs from the specified run
    logs, run = load_logs(project_name, run_id)

    # Print the loaded data (or process it as needed)
    print(logs.keys())
    key1 = "train/losses/log_p_0_T"
    key2 = "eval/losses/log_p_0_T"
    train_measure = add_to_dict(run, key_name = key1)
    test_measure = add_to_dict(run, key_name = key2)

    plt.figure()
    plt.plot(np.arange(0, len(train_measure)), train_measure, "-x", label = "train")
    #plt.plot(np.arange(0, len(test_measure)), test_measure, "-v", label = "val")
    #plt.legend(fontsize = 20)
    plt.ylim(bottom = -50, top = 10)
    plt.xlabel("train steps", fontsize = 20)
    plt.ylabel(r"forward process likelihood", fontsize = 20)
    plt.grid()
    plt.tight_layout()
    plt.savefig("forward_process_likelihood.png", dpi=1200)
    plt.show()


