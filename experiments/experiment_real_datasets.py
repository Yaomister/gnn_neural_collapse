from training import train
from utils import models_list

# Experiment 1: confirming neural collapse happens on real datasets (ENZYMES, MUTAG, PROTEINS)
def run(dataset, dataset_name, model_name, pool, hidden_dim = 512, num_epochs = 3000):
    # the number of classes of the datasets
    num_classes = dataset.num_classes
    # the feature embedding dimensions
    in_dim = dataset.num_node_features

    print(f"training {model_name} on {dataset_name}")

    # load each model
    model = models_list[model_name](
        in_dim = in_dim,
        num_classes = num_classes,
        num_hidden_layers = 3,
        hidden_layer_dim = hidden_dim,
        pool=pool
    )

    # train the model till TPT
    metrics = train(model, graphs=list(dataset), num_classes=num_classes, num_epochs=num_epochs)

    return metrics



    