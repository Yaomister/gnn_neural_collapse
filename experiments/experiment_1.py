
import torch
import numpy as np
from gnns import GCN, GAT, GraphSAGE
from training import train
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset


mutag = TUDataset(root="data/", name= "MUTAG")
proteins = TUDataset(root="data/", name = "PROTEINS")

def run(dataset, dataset_name, hidden_dim = 64, num_epochs = 3000):

    num_classes = dataset.num_classes
    in_dim = dataset.num_node_features

    results = {}

    for model_name, ModelClass in [
        ("GCN", GCN),
        ("GAT", GAT),
        ("GraphSAGE", GraphSAGE)
    ]:
        print(f"training {model_name} on {dataset_name}")

        model = ModelClass(
            in_dim = in_dim,
            num_classes = num_classes,
            num_hidden_layers = 8,
            hidden_layer_dim = hidden_dim
        )

        metrics = train(model, graphs=list(dataset), num_classes=num_classes, num_epochs=num_epochs)

        results[model_name] = metrics

    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for model_name, history in results.items():
        ax[0].plot(history['nc1'], history['epoch'])
        ax[1].plot(history['nc2'], history['epoch'])

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel('NC1')
    ax[0].set_title(f"NC1 over training with {dataset_name}")
    ax[0].legend()

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("NC2")
    ax[1].set_title(f"NC2 over training with {dataset_name}")
    ax[1].legend()

    plt.savefig(f"figures/experiment_1_{dataset_name}.pdf", dpi=300)
    plt.show()

    return results



    





if __name__ == "__main__":
    run(mutag, "MUTAG")
