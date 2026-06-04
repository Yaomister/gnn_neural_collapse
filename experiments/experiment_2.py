from training import train
from utils import models_list
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from sbm import StochasticBlockModel



def run(hidden_dim = 64, num_epochs = 3000):

    results = {}

    num_classes = 3

    # the embedding dimensions of the SBM generated graphs
    in_dim = 16

    homophily = [0.3, 0.6, 0.9]
    noise = [0.3, 0.6, 0.9]

    for model_name, ModelClass in models_list:
        for h in homophily:
            for n in noise:
                
                sbm = StochasticBlockModel(num_nodes=50, num_classes=num_classes, homophily=h, feature_dim=hidden_dim, noise=n)
                model = ModelClass(
                    in_dim = in_dim, 
                    hidden_layer_dim = hidden_dim, 
                    num_classes = num_classes, 
                    num_hidden_layers = 3
                ) 

                graphs = sbm.generate(1000)

                metrics = train(model=model, graphs=graphs, num_classes=3, num_epochs=num_epochs)

                results[f"{model_name}_h{h}_n{n}"] = metrics

    for title, entry in results.items():
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].plot(entry["within_class_variance"], entry["epoch"])
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Within Class Variance")
        axs[0].legend()

        axs[1].plot(entry["class_mean_norms"], entry["epoch"])
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Class Mean Norms")
        axs[1].legend()

        axs[2].plot(entry["class_mean_angles"], entry["epoch"])
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Class Mean Angles")
        axs[2].legend()

        plt.savefig(f"figures/experiment_2_{title}.pdf", dpi=300)        

    return results