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
metrics = ["within_class_variance", "class_mean_angles", "class_mean_norms"]

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
                fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey="row")
                fig.subplots_adjust(wspace=0.05, hspace=0.1)
                for m in range(len(metrics)):
                    for k in range(len(models)):
                        # tag = f"exp1_{models[k]}_ENZYMES_{p.lower()}"
                        tag = f"exp2_{models[k]}_h{h}_n{n}_{p.lower()}"
                        data = group_data(tag, metrics[m])
                        training_accuracy = group_data(tag, "training_accuracy")
                        runs = np.asarray(data)
                        mean = runs.mean(axis=0)
                        epochs = np.arange(1, len(mean) + 1)
                        std = runs.std(axis=0, ddof=1)

                        line,  = ax[m, k].plot(epochs, mean, color="blue")

                        ax[m, k].fill_between(epochs, mean - std, mean + std, color= line.get_color(), alpha=0.2, linewidth=0)
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
                ax[0,0].set_ylim(0.25, 3) 
                ax[1,0].set_ylim(0, 0.6)   
                ax[2,0].set_ylim(0, 0.5) 
                ax[0, 0].set_ylabel("NC1: Within-Class Variance")
                ax[1, 0].set_ylabel("NC2: Class Mean Variance")
                ax[2, 0].set_ylabel("NC2: Class Mean Norms")


                fig.savefig(f"figures/{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)



def plot_experiment_2():
    colors = {50: "red", 20: "green", 10: "blue"} 
    for p in pooling:
        fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
        for m in range(len(metrics)):
            for k in range(len(models)):
                for n in noise:
                    final_metrics, final_metrics_std, tracked_h = [], [], []
                    for h in homophily:
                        tag = f"exp2_{models[k]}_h{h}_n{n}_{p.lower()}"
                        training_accuracy = group_data(tag, "training_accuracy")
                        data = group_data(tag, metrics[m])
                        if not data:
                            continue
                        runs = np.asarray(data)
                        mean = runs.mean(axis=0)
                        std = runs.std(axis=0, ddof=1)
                        

                        tpt = find_tpt(np.asarray(training_accuracy).mean(axis=0))
                        # TPT needs to be reached
                        if tpt is not None:
                            final_metrics.append(mean[-1])
                            final_metrics_std.append(std[-1])
                            tracked_h.append(h)
                    final_metrics = np.asarray(final_metrics)
                    final_metrics_std = np.asarray(final_metrics_std)
                    tracked_h = np.asarray(tracked_h)
                    
                    line, = ax[m, k].plot(tracked_h, final_metrics,
                                          color=colors[n], label=f"noise={n}")

                    
                    lower = np.clip(np.minimum(final_metrics_std, final_metrics - 1e-9), 0, None)
                    upper = final_metrics_std  

                    ax[m, k].errorbar(tracked_h, final_metrics, yerr=[lower, upper], color=colors[n], capsize=3, linewidth=2, elinewidth=1, label=f"noise={n}")
                    if m == 0:
                        ax[m, k].set_title(models[k])
                    ax[m, k].yaxis.set_major_locator(ticker.LogLocator(base=10))
                    ax[m, k].yaxis.set_minor_locator(ticker.NullLocator())  
                    ax[2, k].set_xlabel("Graph Homophily")
                    ax[m, k].xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax[m,0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax[m,0].yaxis.set_minor_locator(ticker.NullLocator())
            ax[m,0].set_ylabel(metrics[m].replace("_"," "))
        ax[0, 0].set_ylabel("NC1: Within-Class Variance")
        ax[1, 0].set_ylabel("NC2: Class Mean Variance")
        ax[2, 0].set_ylabel("NC2: Class Mean Norms")
        ax[0, 2].legend(loc="upper right")
        fig.savefig(f"figures/{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)







if __name__ == "__main__":
    plot_experiment_1()
    # plot_experiment_2()
    



# Plot 3: within-class and between class-energy vs epoch time (same grpah)
# Plot 4: NC1 and NC2 metrics plotted against training epoch, one line per pooling operator 