
import numpy as np
from training import train
from utils import models_list
import matplotlib.pyplot as plt

def run(dataset, dataset_name, hidden_dim = 64, num_epochs = 3000):

    num_classes = dataset.num_classes
    in_dim = dataset.num_node_features

    results = {}

    for model_name, ModelClass in models_list:
        for pool in ["max", "mean"]:
            model_name = f"{model_name}_{pool}"
            print(f"training {model_name} on {dataset_name}")

            model = ModelClass(
                in_dim = in_dim,
                num_classes = num_classes,
                num_hidden_layers = 3,
                hidden_layer_dim = hidden_dim,
                pool=pool
            )

            metrics = train(model, graphs=list(dataset), num_classes=num_classes, num_epochs=num_epochs)

            results[model_name] = metrics

        # plotting the results
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        for model_name, history in results.items():
            ax[0].plot(history['within_class_variance'], history['epoch'])
            ax[1].plot(history['class_mean_norms'], history['epoch'])
            ax[2].plot(history['class_mean_angles'], history['epoch'])


        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel('Within-Class Variance')
        ax[0].set_title(f"Within-Class Variance (NC1) over training with {dataset_name}")
        ax[0].legend()

        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Class Mean Norms")
        ax[1].set_title(f"Class Mean Norms (NC2) over training with {dataset_name}")
        ax[1].legend()

        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Class Mean Angles")
        ax[2].set_title(f"Class Mean Angles (NC2) over training with {dataset_name}")
        ax[2].legend()

        plt.savefig(f"figures/experiment_1_{dataset_name}.pdf", dpi=300)
        plt.show()

        return results



    