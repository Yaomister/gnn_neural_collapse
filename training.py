import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from nc import calculate_metrics



def train(model, graphs, num_classes, num_epochs, learning_rate = 1e-4, weight_decay = 1e-4, measure_interval = 50):

    loader = DataLoader(dataset = graphs, batch_size=32, shuffle=True)

    optimizer = Adam(
        model = model.parameters(),
        lr = learning_rate,
        weight_decay=weight_decay
    )

    record = {
        "epoch" : [],
        "nc1": [],
        "nc2" : [],
        "training_loss": [],
        "training_accuracy": []
    }
    

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total = 0
        correct = 0

        for batch in loader:
            optimizer.zero_grad()
            logits, graph_representation = model(batch.x, batch.edge_index, batch.batch)

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

            metrics = calculate_metrics(all_representation, all_true_labels, num_classes)

            record["epoch"].append(epoch)
            record["nc1"].append(metrics["nc1"])
            record['nc2'].append(metrics['nc2'])
            record['training_losss'].append(total_loss/ len(loader))
            record['training_accuracy'].append(correct/total)

            print(f"epoch : {epoch:4d} | loss : {total_loss / len(loader):.4f} | accuracy : {correct/total:.3f} | NC1 : {record["nc1"]:.4f} | NC2 : {record["nc2"]:.4f}")

    return record


def _measure_neural_collapse(model, graphs):

    loader = DataLoader(dataset=graphs, batch_size=64, shuffle=False)
    
    all_graph_representations = []
    all_true_labels = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            _, graph_representation = model(batch.x, batch.edge_index, batch.batch)
            all_graph_representations.append(graph_representation)
            all_true_labels.append(batch.y)

    model.train()
    return torch.cat(all_graph_representations), torch.cat(all_true_labels)

