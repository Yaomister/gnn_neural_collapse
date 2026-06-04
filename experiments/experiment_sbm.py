from training import train
from utils import models_list
import matplotlib.pyplot as plt
from sbm import StochasticBlockModel



def run(hidden_dim = 64, num_epochs = 3000):

    results = {}

    # the embedding dimensions of the SBM generated graphs
    in_dim = 16
    
    homophily = [0.3, 0.6, 0.9]
    noise = [0.3, 0.6, 0.9]

    for model_name, ModelClass in models_list:
         for pool in ["max", "mean"]:
            model_name = f"{model_name}_{pool}"
            for h in homophily:
                for n in noise:

                    sbm = StochasticBlockModel(num_nodes=50, num_graph_classes=3, num_node_classes=3, homophily=h, feature_dim=in_dim, noise=n)
                    model = ModelClass(
                        in_dim = in_dim, 
                        hidden_layer_dim = hidden_dim, 
                        num_classes = 3, 
                        num_hidden_layers = 3,
                        pool= pool
                    ) 

                    graphs = sbm.generate(1000)

                    metrics = train(model=model, graphs=graphs, num_classes=3, num_epochs=num_epochs)

                    results[f"{model_name}_h{h}_n{n}"] = metrics

    for title, entry in results.items():
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].plot( entry["epoch"], entry["within_class_variance"])
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Within Class Variance")
        axs[0].legend()

        axs[1].plot(entry["epoch"],entry["class_mean_norms"])
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Class Mean Norms")
        axs[1].legend()

        axs[2].plot(entry["epoch"], entry["class_mean_angles"])
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Class Mean Angles")
        axs[2].legend()

        plt.savefig(f"figures/experiment_2_{title}.pdf", dpi=300) 

        energies = entry["dirichlet_energies_at_intermediate_layers"]
        epochs = entry["epoch"]

        num_layers = len(energies[0])

        fig, axs = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))

        for l in range(num_layers):
            axs[l].plot(entry["epoch"], [energies[e][l][0] for e in range(len(epochs))], label = "within-class")
            axs[l].plot(entry["epoch"], [energies[e][l][1] for e in range(len(epochs))], label = "between-class")
            axs[list].set_xlabel("Epoch")
            axs[l].set_ylabel("Dirichlet energy")
            axs[l].set_title(f"Layer {l+1}")
            axs[l].legend()
        plt.savefig(f"figures/experiment_3_energy_{title}.pdf", dpi=300)
        plt.close()

    return results





