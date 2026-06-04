
import os
import torch
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)

def load_results():
    results = {}
    for file_name in os.listdir("results"):
        if file_name.endswith(".pt"):
            tag = file_name[:-3]
            results[tag] = torch.load(f"results/{tag}.pt")
    return results

def plot_nc(tag, record):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    epochs = record["epoch"]

    axs[0].plot(epochs, record["within_class_variance"])
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("NC1: within-class variance")
    axs[0].set_yscale("log")

    axs[1].plot(epochs, record["class_mean_norms"])
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("NC2: class-mean norm std")

    axs[2].plot(epochs, record["class_mean_angles"])
    axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("NC2: equiangularity")

    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_nc.pdf", dpi=300)
    plt.close()

def plot_energy(tag, record):
    energies = record["dirichlet_energies_at_intermediate_layers"] 
    epochs = record["epoch"]
    num_layers = len(energies[0])

    fig, axs = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    for l in range(num_layers):
        within  = [energies[e][l][0] for e in range(len(epochs))]
        between = [energies[e][l][1] for e in range(len(epochs))]
        axs[l].plot(epochs, within,  label="within-class")
        axs[l].plot(epochs, between, label="between-class")
        axs[l].set_xlabel("Epoch"); axs[l].set_ylabel("Dirichlet energy")
        axs[l].set_yscale("log")
        axs[l].set_title(f"Layer {l+1}"); axs[l].legend()
    fig.suptitle(tag)
    plt.savefig(f"figures/{tag}_energy.pdf", dpi=300)
    plt.close()



if __name__ == "__main__":
    results = load_results()

    for tag, record in results.items():
        plot_nc(tag, record)
        if "dirichlet_energies_at_intermediate_layers" in record:
            plot_energy(tag, record)

