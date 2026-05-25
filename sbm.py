import torch
import numpy as np
from torch_geometric.data import Data

class StochasticBlockModel():

    def __init__(self, num_nodes, num_classes,  homophily, feature_dim, density = 0.3, noise = 0.1):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.homophily = homophily
        self.feature_dim = feature_dim
        self.density = density
        self.noise = noise
        # treating the fraction of same class pairs as 1 instead of 1/c (shouldn't matter)

        self.p_in = homophily * density
        # solved out based on the equation density == p_in + (c - 1)p_out
        self.p_out =   (1 - homophily) * density / (num_classes - 1) if num_classes > 1 else 0

        self.nodes_per_class = num_nodes // num_classes

        
    
    def generate(self, num_graphs):
        graphs = []
        
        for graph_id in range(num_graphs):
            node_labels = torch.zeros(self.num_nodes, dtype=torch.long)

            for c in range(self.num_classes):
                start = c * self.num_classes
                end = ((c + 1) * self.num_classes) if c < self.num_classes - 1 else self.num_nodes
                node_labels[start:end] = c


            edges = []

            for u in range(self.num_nodes):
                for v in range(u + 1, self.num_nodes):
                    if (node_labels[u] == node_labels[v]):
                        prob = self.p_in
                    else:
                        prob = self.p_out
                    if (np.random.randint() < prob):
                        edges.append([u, v])
                        edges.append([v, u])

            if len(edges) == 0:
                edges = [[0, 0]]

            edge_index = torch.tensor(edges, dtype=torch.long)

            class_means = torch.randn(self.num_classes, self.feature_dim)

            x = torch.zeros(self.num_nodes, self.feature_dim)
            for c in range(self.num_classes):
                mask = (node_labels == c)
                x[mask] = class_means[c] + self.noise * torch.random(mask.sum(), self.feature_dim)

            graph_label = graph_id % self.num_classes


            graphs.append(Data(x=x, y = torch.tensor([graph_label]), edge_index = edge_index, node_labels=node_labels))

        return graphs



