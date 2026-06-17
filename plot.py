import os
import torch
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)

homophily = [0.1, 0.3, 0.5, 0.7, 0.9]
pooling = ["max", "mean", "sum"]
noise = [50, 20, 10]
models = ["GCN", "GAT", "GraphSAGE"]
nc_metrics = ["within_class_variance", "class_mean_angles", "class_mean_norms"]

# load all the .pt files containing the data
def load_results():
    results = {}
    for file_name in os.listdir("results"):
        if file_name.endswith(".pt"):
            tag = file_name[:-3]
            results[tag] = torch.load(f"results/{tag}.pt", weights_only=False)
    return results



def find_tpt(accuracy, threshold = 0.99, patience = 5):
    streak = 0

    for i, acc in enumerate(accuracy):
        streak = streak + 1 if acc > threshold else 0
        if streak > patience:
            return i - patience + 1
        
    return None
        

def add_tpt_line(ax, tpt):
    if tpt is not None:
        ax.axvline(x = tpt, color = "red", linestyle="--", linewidth=1.5, label="TPT")

        

def group_data(tag, key):
    data = []
    results = load_results()
    for i in range(5):
        i_tag = f"{tag}_{i}"  
        if i_tag in results:
            data.append(results[i_tag][key])

    return data


# Plot 1: NC1 and NC2 metrics plotted against training epoch for each GNN architecutre (for synthetic datasets and ENZYMES)
def plot_experiment_1():
    for h in homophily:
        for n in noise:
            for p in pooling:
                fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")
                fig.subplots_adjust(wspace=0.05, hspace=0.1)
                for m in range(len(nc_metrics)):
                    for k in range(len(models)):
                        tag = f"exp1_{models[k]}_ENZYMES_{p.lower()}"
                        # tag = f"exp2_{models[k]}_h{h}_n{n}_{p.lower()}"
                        data = group_data(tag, nc_metrics[m])
                        training_accuracy = group_data(tag, "training_accuracy")
                        runs = np.asarray(data)
                        mean = runs.mean(axis=0)
                        epochs = np.arange(1, len(mean) + 1)
                        std = runs.std(axis=0, ddof=1)

                        ax[m, k].plot(epochs, mean, color="blue")

                        ax[m, k].fill_between(epochs, mean - std, mean + std, color= "blue", alpha=0.2, linewidth=0)
                        # ax[m, k].set_yscale("log")
                        # ax[m, k].set_ylim(0, 1.5)
                        if m == 0:
                            ax[m, k].set_title(models[k])

                        ax[2, k].set_xlabel("Epoch")
                        
                        tpt = find_tpt(np.asarray(training_accuracy).mean(axis=0))
                        add_tpt_line(ax[m, k], tpt)
                    ax[m, 0].xaxis.set_major_locator(ticker.MultipleLocator(10))
                    ax[m, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * 5)}"))
                    ax[m,0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax[m,0].yaxis.set_minor_locator(ticker.NullLocator())
                    
                ax[0, 2].legend(loc="upper right")
                ax[0,0].set_ylim(0, 8) 
                ax[1,0].set_ylim(0, 0.6)   
                ax[2,0].set_ylim(0, 0.5) 
                ax[0, 0].set_ylabel("NC1: Within-Class Variance")
                ax[1, 0].set_ylabel("NC2: Class Mean Variance")
                ax[2, 0].set_ylabel("NC2: Class Mean Norms")


                fig.savefig(f"figures/graph_1_h{h}_n{n}_{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)



def plot_experiment_2():
    colors = {50: "red", 20: "green", 10: "blue"} 
    for p in pooling:
        fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
        for m in range(len(nc_metrics)):
            for k in range(len(models)):
                for n in noise:
                    final_metrics, final_metrics_std, tracked_h = [], [], []
                    for h in homophily:
                        tag = f"exp2_{models[k]}_h{h}_n{n}_{p.lower()}"
                        training_accuracy = group_data(tag, "training_accuracy")
                        data = group_data(tag, nc_metrics[m])
                        if not data:
                            continue
                        runs = np.asarray(data)
                        mean = runs.mean(axis=0)
                        std = runs.std(axis=0, ddof=1)
                        

                        tpt = find_tpt(np.asarray(training_accuracy).mean(axis=0))
                        # TPT needs to be reached
                        # if tpt is not None:
                        final_metrics.append(mean[-1])
                        final_metrics_std.append(std[-1])
                        tracked_h.append(h)
                    final_metrics = np.asarray(final_metrics)
                    final_metrics_std = np.asarray(final_metrics_std)
                    tracked_h = np.asarray(tracked_h)
                    
                    ax[m, k].plot(tracked_h, final_metrics, color=colors[n], label=f"noise={n}")
                    ax[m, k].fill_between(tracked_h,
                                        final_metrics - final_metrics_std,
                                        final_metrics + final_metrics_std,
                                        color=colors[n], alpha=0.2)
                    if m == 0:
                        ax[m, k].set_title(models[k])
                    ax[m, k].yaxis.set_major_locator(ticker.LogLocator(base=10))
                    ax[m, k].yaxis.set_minor_locator(ticker.NullLocator())  
                    ax[2, k].set_xlabel("Graph Homophily")
                    ax[m, k].xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax[m,0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax[m,0].yaxis.set_minor_locator(ticker.NullLocator())
            ax[m,0].set_ylabel(nc_metrics[m].replace("_"," "))
        ax[0, 0].set_ylabel("NC1: Within-Class Variance")
        ax[1, 0].set_ylabel("NC2: Class Mean Variance")
        ax[2, 0].set_ylabel("NC2: Class Mean Norms")
        ax[0, 2].legend(loc="upper right")
        fig.savefig(f"figures/graph_2_p{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


def plot_experiment_3():
    for n in noise:
        fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey="row")
        for i, p in enumerate(pooling):
            for k, m in enumerate(models):
                final_within_class_energy_means, final_between_class_energy_means, final_within_class_energy_std, final_between_class_energy_std = [], [], [], []
                for h in homophily:
                    tag = f"exp2_{m}_h{h}_n{n}_{p.lower()}"
                    data = group_data(tag, "dirichlet_energies_at_intermediate_layers")
                    within_class_energy = [[e[-1][0] for e in seed_run][-1] for seed_run in data]
                    between_class_energy = [[e[-1][1] for e in seed_run][-1] for seed_run in data]
                    if not data:
                        continue
                    within_class_energy_runs = np.asarray(within_class_energy)
                    within_class_energy_mean = within_class_energy_runs.mean(axis=0)
                    within_class_energy_std = within_class_energy_runs.std(axis=0, ddof=1)
                    
                    between_class_energy_runs = np.asarray(between_class_energy)
                    between_class_energy_mean = between_class_energy_runs.mean(axis=0)
                    between_class_energy_std = between_class_energy_runs.std(axis=0, ddof=1)


                    final_within_class_energy_means.append(within_class_energy_mean)
                    final_between_class_energy_means.append(between_class_energy_mean)
                    final_between_class_energy_std.append(between_class_energy_std)
                    final_within_class_energy_std.append(within_class_energy_std)

                final_between_class_energy_means = np.asarray(final_between_class_energy_means)
                final_between_class_energy_std = np.asarray(final_between_class_energy_std)
                final_within_class_energy_means = np.asarray(final_within_class_energy_means)
                final_within_class_energy_std = np.asarray(final_within_class_energy_std)

                ax[i, k].plot(homophily, final_between_class_energy_means, color="blue", label = "Between-Class Energy")
                ax[i, k].fill_between(homophily, final_between_class_energy_means - final_between_class_energy_std, final_between_class_energy_means + final_between_class_energy_std, color= "blue", alpha=0.2, linewidth=0)
                ax[i, k].plot(homophily, final_within_class_energy_means, color="green", label = "Within-Class Energy")
                ax[i, k].fill_between(homophily, final_within_class_energy_means - final_within_class_energy_std, final_within_class_energy_means + final_within_class_energy_std, color= "green", alpha=0.2, linewidth=0)
                ax[i, k].yaxis.set_major_locator(ticker.LogLocator(base=10))
                ax[i, k].yaxis.set_minor_locator(ticker.NullLocator())  
                ax[i, k].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax[2, k].set_xlabel("Graph Homophily")
            ax[i, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * 5)}"))
            ax[i,0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax[i,0].yaxis.set_minor_locator(ticker.NullLocator())
        ax[0, 2].legend(loc="upper right")
        ax[0, 0].set_ylabel("Max-Pool")
        ax[1, 0].set_ylabel("Mean-Pool")
        ax[2, 0].set_ylabel("Sum-Pool")
        fig.savefig(f"figures/graph_3_n{n}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

def plot_experiment_4():
    colors = {"max": "red", "sum": "green", "mean": "blue"} 
    for n in noise:
        for h in homophily:
            fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")
            for i, metric in enumerate(nc_metrics):
                for j, m in enumerate(models):
                    for p in pooling:
                        tag = f"exp2_{m}_h{h}_n{n}_{p.lower()}"
                        data = group_data(tag, metric)
                        
                        runs = np.asarray(data)
                        mean = runs.mean(axis=0)
                        std = runs.std(axis = 0, ddof=1)
                        training_accuracy = group_data(tag, "training_accuracy")
                        tpt = find_tpt(np.asarray(training_accuracy).mean(axis=0))
                        if tpt is None:
                            continue

                        mean_to_plot = mean[tpt:]
                        mean_to_plot = mean[tpt:]
                        epochs = np.arange(tpt, tpt + len(mean_to_plot))   # was: np.arange(1, len+1)
                        ax[i, j].fill_between(epochs, mean_to_plot - std[tpt:], mean_to_plot + std[tpt:],alpha=0.2, linewidth=0, color=colors[p])
                        if i == 0:
                            ax[i, j].set_title(m)
                        ax[i, j].plot(epochs, mean_to_plot, color=colors[p], label = f"{p} pool")
                ax[i,0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax[i,0].yaxis.set_minor_locator(ticker.NullLocator())
                ax[i, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * 5)}"))
            ax[0, 0].set_ylabel("NC1: Within-Class Variance")
            ax[1, 0].set_ylabel("NC2: Class Mean Variance")
            ax[2, 0].set_ylabel("NC2: Class Mean Norms")
            ax[0, 2].legend(loc="upper right")
            fig.subplots_adjust(hspace=0.1)
            fig.savefig(f"figures/graph_4_n{n}_h{h}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)



                        
                    


if __name__ == "__main__":
    # plot_experiment_1()
    # plot_experiment_2()
    # plot_experiment_3()
    plot_experiment_4()