
import numpy as np
from training import train
from utils import models_list
import matplotlib.pyplot as plt

def run(dataset, dataset_name, model_name, pool, hidden_dim = 64, num_epochs = 3000):

    num_classes = dataset.num_classes
    in_dim = dataset.num_node_features

    results = {}
    print(f"training {model_name} on {dataset_name}")

    model = models_list[model_name](
        in_dim = in_dim,
        num_classes = num_classes,
        num_hidden_layers = 3,
        hidden_layer_dim = hidden_dim,
        pool=pool
    )

    model_name = f"{model_name}_{pool}_{dataset_name}"

    metrics = train(model, graphs=list(dataset), num_classes=num_classes, num_epochs=num_epochs)

    results[model_name] = metrics

    return results



    