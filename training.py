import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from nc import calculate_metrics
from dirichlet_energy import calculate_dirichlet_energy


def train(model, graphs, num_classes, num_epochs, measure_energy=False, learning_rate=1e-3, weight_decay=1e-5, measure_interval=50):
    """Train a GNN for graph classification and record metrics.

    Args:
        model: GNN with forward(x, edge_index, batch) -> (logits, graph_repr, layer_list).
        graphs: List of PyG Data objects with x, edge_index, y, and (if we're measuring the Dirichlet energy) node_labels.
        num_classes: Number of graph classes.
        num_epochs: Total training epochs.
        measure_energy: Whether to compute per-layer Dirichlet energy (only for SBM graphs that carry node_labels; requires batch_size=1 to avoid cross-graph contamination).
        measure_interval: How often (in epochs) to log metrics.

    Returns:
        dict with per-measurement-epoch lists for loss, accuracy, and NC metrics.
    """

    # move everything to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("RUNNING ON CPU", flush=True)

    # load the training dataset
    loader = DataLoader(dataset = graphs, batch_size=32, shuffle=True)

    # define ADAM optimizer with learning rate and weight decay
    optimizer = Adam(
        params = model.parameters(),
        lr = learning_rate,
        weight_decay=weight_decay
    )

    record = {
        "epoch" : [],
        "within_class_variance": [],
        "class_mean_norms" : [],
        "class_mean_angles" : [],
        "last_layer_node_within_class_variance": [],
        "last_layer_node_class_mean_norms" : [],
        "last_layer_node_class_mean_angles" : [],
        "training_loss": [],
        "training_accuracy": [],
    }

    model.to(device)
    model.train()

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total = 0
        correct = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, _, _ = model(batch.x, batch.edge_index, batch.batch)

            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()

            # keep track of the loss and of the number of correct predictions
            total_loss += loss.item()
            
            prediction = logits.argmax(dim= 1)
            
            # keep track of training loss and training accuracy
            total += batch.y.size(0)
            correct += (prediction == batch.y).sum().item()

        # when we want to log the training statistics
        if (epoch % measure_interval == 0):
            all_representation, all_true_labels, all_last_layer_node_features, all_last_layer_node_labels = _measure_neural_collapse(model, graphs)

            # calculate NC metrics
            nc1, nc2 = calculate_metrics(all_representation, all_true_labels, num_classes)
            node_nc1, node_nc2 = calculate_metrics(all_last_layer_node_features, all_last_layer_node_labels, num_classes=3)
            # when we want to measure the Dirichlet energy (only for the synthetic graphs)
            if (measure_energy):
                if "dirichlet_energies_at_intermediate_layers" not in record:
                    record["dirichlet_energies_at_intermediate_layers"] = []
                dirichlet_energies_at_intermediate_layers = _measure_dirichlet_energy(model, graphs, num_layers=len(model.layers))
                record['dirichlet_energies_at_intermediate_layers'].append(dirichlet_energies_at_intermediate_layers)

            record["epoch"].append(epoch)
            record["within_class_variance"].append(nc1["within_class_variance"])
            record['class_mean_norms'].append(nc2['class_mean_norms'])
            record['class_mean_angles'].append(nc2['class_mean_angles'])
            record["last_layer_node_within_class_variance"].append(nc1["within_class_variance"])
            record['last_layer_node_class_mean_norms'].append(nc2['class_mean_norms'])
            record['last_layer_node_class_mean_angles'].append(nc2['class_mean_angles'])
            record['training_loss'].append(total_loss/ len(loader))
            record['training_accuracy'].append(correct/total)
            
            # print how its doing
            print(f"epoch : {epoch:4d} | loss : {total_loss / len(loader):.4f} | accuracy : {correct/total:.3f} | within class variance : {record['within_class_variance'][-1]:.4f} | class mean norms : {record['class_mean_norms'][-1]:.4f} | class mean angles : {record['class_mean_angles'][-1]:.4f}")

    return record

def _measure_neural_collapse(model, graphs):
    """Collect pooled graph representations and last-layer node features across the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset=graphs, batch_size=64, shuffle=False)
    
    all_graph_representations = []
    all_true_labels = []
    all_last_layer_node_features = []
    all_last_layer_node_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, graph_representation, intermediate_layers = model(batch.x, batch.edge_index, batch.batch)
            node_features = intermediate_layers[-1] 
            graph_label_per_node = batch.y[batch.batch] 

            all_last_layer_node_features.append(node_features)
            all_last_layer_node_labels.append(graph_label_per_node)

            all_graph_representations.append(graph_representation)
            all_true_labels.append(batch.y)

    model.train()
    return torch.cat(all_graph_representations), torch.cat(all_true_labels), torch.cat(all_last_layer_node_features), torch.cat(all_last_layer_node_labels)


def _measure_dirichlet_energy(model, graphs, num_layers):
    """Compute average within/between-class Dirichlet energy per layer across the dataset.

    Uses batch_size=1 so each graph's edge_index refers only to its own nodes; batching
    multiple graphs would merge them into one disconnected graph and misalign node_labels.
    Returns a list of [within_avg, between_avg] for each GNN layer.
    """
    # we have to use batches of 1, or else everything gets concatnated into one big graph, which messes up the split between within-class and between-class energies
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset=graphs, batch_size=1, shuffle=False)

    within_totals = [0.0] * num_layers
    between_totals = [0.0] * num_layers

    n = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, intermediate_layers = model(batch.x, batch.edge_index, batch.batch)
            for l, layer in enumerate(intermediate_layers):
                w, b = calculate_dirichlet_energy(layer, batch.edge_index, batch.node_labels)
                within_totals[l] += w.item()
                between_totals[l] += b.item()
            n += 1
    model.train()

    return [[within_totals[l]/n, between_totals[l]/n] for l in range(num_layers)]
                                      