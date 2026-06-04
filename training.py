import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from nc import calculate_metrics
from dirichlet_energy import calculate_dirichlet_energy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# following the learning rate defined in the paper
def train(model, graphs, num_classes, num_epochs, learning_rate = 1e-3, weight_decay = 1e-3, measure_interval = 50):

    loader = DataLoader(dataset = graphs, batch_size=32, shuffle=True)

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
        "training_loss": [],
        "training_accuracy": [],
        "dirichlet_energies_at_intermediate_layers": []
    }

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total = 0
        correct = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, graph_representation, intermediate_layers = model(batch.x, batch.edge_index, batch.batch)

            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()

            # keep track of the loss and of the number of correct predictions
            total_loss += loss.item()
            
            prediction = logits.argmax(dim= 1)

            total += batch.y.size(0)
            correct += (prediction == batch.y).sum().item()

        if (epoch % measure_interval == 0):
            all_representation, all_true_labels = _measure_neural_collapse(model, graphs)

            nc1, nc2 = calculate_metrics(all_representation, all_true_labels, num_classes)

            dirichlet_energies_at_intermediate_layers = _measure_dirichlet_energy(model, graphs, num_layers=len(model.layers))

            record["epoch"].append(epoch)
            record["within_class_variance"].append(nc1["within_class_variance"])
            record['class_mean_norms'].append(nc2['class_mean_norms'])
            record['class_mean_angles'].append(nc2['class_mean_angles'])
            record['training_loss'].append(total_loss/ len(loader))
            record['training_accuracy'].append(correct/total)
            record['dirichlet_energies_at_intermediate_layers'].append(dirichlet_energies_at_intermediate_layers)

            print(f"epoch : {epoch:4d} | loss : {total_loss / len(loader):.4f} | accuracy : {correct/total:.3f} | within class variance : {record['within_class_variance'][-1]:.4f} | class mean norms : {record['class_mean_norms'][-1]:.4f} | class mean angles : {record['class_mean_angles'][-1]:.4f}")

    return record


def _measure_neural_collapse(model, graphs):

    loader = DataLoader(dataset=graphs, batch_size=64, shuffle=False)
    
    all_graph_representations = []
    all_true_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, graph_representation, _ = model(batch.x, batch.edge_index, batch.batch)
            all_graph_representations.append(graph_representation)
            all_true_labels.append(batch.y)

    model.train()
    return torch.cat(all_graph_representations), torch.cat(all_true_labels)



def _measure_dirichlet_energy(model, graphs, num_layers):

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
                                      