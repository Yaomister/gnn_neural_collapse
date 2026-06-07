import os
import torch
import numpy as np
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)

# load all the .pt files containing the data
def load_results():
    results = {}
    for file_name in os.listdir("results"):
        if file_name.endswith(".pt"):
            tag = file_name[:-3]
            results[tag] = torch.load(f"results/{tag}.pt", weights_only=False)
    return results

# find the epoch where the terminal phase of training (TPT) begins
def find_tpt(record, threshold=0.95, patience=5):
    accs = record["training_accuracy"]
    epochs = record["epoch"]
    streak = 0
    # as long as the training accuracy is above 0.95 for the past 5 epochs, we say that TPT has started
    for i, a in enumerate(accs):
        streak = streak + 1 if a >= threshold else 0
        if streak >= patience:
            return epochs[i - patience + 1]
    return None

# for comparing graphs side by side
def plot_nc_overlay(records_by_label, metric, ylabel, fname, logy=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, record in records_by_label.items():
        ax.plot(record["epoch"], record[metric], label=label)
    if logy: ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
    fig.savefig(f"figures/{fname}.png"); plt.close()


# add a red line to indicte when TPT has started
def add_tpt_line(ax, tpt):
    if tpt is not None:
        ax.axvline(x=tpt, color="red", linestyle="--", linewidth=1.5, label="TPT")


# plot the within-class variance, norm standard deviation, and equitriangularity
def plot_nc(tag, record):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    epochs = record["epoch"]
    tpt = find_tpt(record)

    axs[0].plot(epochs, record["within_class_variance"])
    # axs[0].set_yscale("log")
    axs[0].set_ylabel("NC1: within-between-class variance ratio")
    axs[1].plot(epochs, record["class_mean_norms"]); axs[1].set_ylabel("NC2: norm standard deviation")
    axs[2].plot(epochs, record["class_mean_angles"]); axs[2].set_ylabel("NC2: equiangularity")

    for ax in axs:
        ax.set_xlabel("Epoch")
        add_tpt_line(ax, tpt)
        ax.legend()
    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_nc.png", dpi=300); plt.close()


# plot the Dirichlet energy
def plot_energy(tag, record):
    # we're not tracking the dirichlet energy for the real graphs
    energies = record["dirichlet_energies_at_intermediate_layers"]
    if not energies:
        return
    epochs = record["epoch"]
    tpt = find_tpt(record)
    num_layers = len(energies[0])

    fig, axs = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    for l in range(num_layers):
        within  = [energies[e][l][0] for e in range(len(epochs))]
        between = [energies[e][l][1] for e in range(len(epochs))]
        axs[l].plot(epochs, within,  label="within-class")
        axs[l].plot(epochs, between, label="between-class")
        # axs[l].set_yscale("log")
        axs[l].set_xlabel("Epoch")
        axs[l].set_title(f"Layer {l+1}")
        add_tpt_line(axs[l], tpt)
        axs[l].legend()

    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_energy.png", dpi=300); plt.close()

# plot the training loss and accuracy
def plot_training(tag, record):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    epochs = record["epoch"]
    tpt = find_tpt(record)

    axs[0].plot(epochs, record["training_loss"]);     axs[0].set_ylabel("Training loss")
    axs[1].plot(epochs, record["training_accuracy"]); axs[1].set_ylabel("Training accuracy")

    for ax in axs:
        ax.set_xlabel("Epoch")
        add_tpt_line(ax, tpt)
        ax.legend()
    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_training.png", dpi=300); plt.close()




def plot_figure_2(results):
    models = ["GCN", "GAT", "GraphSAGE"]
    homophily = [0.3, 0.6, 0.9]
    noise = [50, 20, 10]        # high -> low
    p = 'mean'                  # your claim is about mean pooling

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for j, n in enumerate(noise):
        for m in models:
            means, stds = [], []
            for h in homophily:
                value = results[f"exp2_{m}_h{h}_n{n}_{p}"]["within_class_variance"][-10:]
                mu, sd = np.mean(value), np.std(value)
                means.append(mu); stds.append(sd)
            axs[j].errorbar(homophily, means, yerr=stds, marker='o', capsize=3, label=m)
        axs[j].set_title(f"noise = {n}")
        axs[j].set_xlabel("homophily")
        axs[j].set_ylabel(r"$\mathcal{NC}1$ floor")
        axs[j].legend()
    fig.savefig(f"figures/fig_homophily_{p}.png", dpi=300)
    plt.close()

   

def plot_figure_1(results):
    fig, axs = plt.subplots(2, 3, figsize=(25, 10), sharey="col")
    models = [
        {"GCN (mean pool)": "exp1_GCN_ENZYMES_mean",
        "GAT (mean pool)": "exp1_GAT_ENZYMES_mean",
        "GraphSAGE (mean pool)": "exp1_GraphSAGE_ENZYMES_mean"},
        {
        "GCN (max pool)": "exp1_GCN_ENZYMES_max",
        "GAT (max pool)": "exp1_GAT_ENZYMES_max",
        "GraphSAGE (max pool)": "exp1_GraphSAGE_ENZYMES_max",
    }
    ]

    metrics = [
        ("within_class_variance", "NC1: within-class variance"),
        ("class_mean_angles", "NC2: equiangularity"),
        ("class_mean_norms", "NC2: norm std")
    ]

    for i, model in enumerate(models):
        for ax, (metric, ylabel) in zip(axs[i], metrics):
            for label, tag in model.items():
                r = results[tag]
                ax.plot(r['epoch'], r[metric], label=label)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
        
        fig.savefig("figures/fig1.png", dpi=300); 
    
    plt.close()
    
if __name__ == "__main__":
    results = load_results()

    # plot everything
    # for tag, record in results.items():
    #     plot_nc(tag, record)
    #     plot_training(tag, record)
    #     if record.get("dirichlet_energies_at_intermediate_layers"):
    #         plot_energy(tag, record)

    plot_figure_1(results)
    plot_figure_2(results)
 




    

