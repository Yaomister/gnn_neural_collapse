
import os
import torch
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)

def load_results():
    results = {}
    for file_name in os.listdir("results"):
        if file_name.endswith(".pt"):
            tag = file_name[:-3]
            results[tag] = torch.load(f"results/{tag}.pt", weights_only=False)
    return results

def find_tpt(record, threshold=0.95, patience=5):
    accs = record["training_accuracy"]
    epochs = record["epoch"]
    streak = 0
    for i, a in enumerate(accs):
        streak = streak + 1 if a >= threshold else 0
        if streak >= patience:
            return epochs[i - patience + 1]
    return None

def plot_nc_overlay(records_by_label, metric, ylabel, fname, logy=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, record in records_by_label.items():
        ax.plot(record["epoch"], record[metric], label=label)
    if logy: ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
    fig.savefig(f"figures/{fname}.pdf"); plt.close()


def add_tpt_line(ax, tpt):
    if tpt is not None:
        ax.axvline(x=tpt, color="red", linestyle="--", linewidth=1.5, label="TPT")

def plot_nc(tag, record):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    epochs = record["epoch"]
    tpt = find_tpt(record)

    axs[0].plot(epochs, record["within_class_variance"]); axs[0].set_yscale("log")
    axs[0].set_ylabel("NC1: within-class variance")
    axs[1].plot(epochs, record["class_mean_norms"]); axs[1].set_ylabel("NC2: norm std")
    axs[2].plot(epochs, record["class_mean_angles"]); axs[2].set_ylabel("NC2: equiangularity")

    for ax in axs:
        ax.set_xlabel("Epoch")
        add_tpt_line(ax, tpt)
        ax.legend()
    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_nc.pdf", dpi=300); plt.close()


def plot_energy(tag, record):
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
        axs[l].set_yscale("log"); axs[l].set_xlabel("Epoch")
        axs[l].set_title(f"Layer {l+1}")
        add_tpt_line(axs[l], tpt)
        axs[l].legend()
    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_energy.pdf", dpi=300); plt.close()


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
    plt.savefig(f"figures/{tag}_training.pdf", dpi=300); plt.close()

if __name__ == "__main__":
    results = load_results()

    for tag, record in results.items():
        plot_nc(tag, record)
        plot_training(tag, record)
        if record.get("dirichlet_energies_at_intermediate_layers"):
            plot_energy(tag, record)

    plot_nc_overlay(
    {"mean": results["exp1_GCN_ENZYMES_mean"],
     "max":  results["exp1_GCN_ENZYMES_max"]},
    metric="within_class_variance", ylabel="NC1", fname="exp1_GCN_pool_compare", logy=True,
    )

