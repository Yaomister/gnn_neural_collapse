import torch
import numpy as np
from torch_geometric.data import Data


composition = torch.tensor([
    [0.10, 0.10, 0.80],  
    [0.30, 0.30, 0.40],  
    [0.50, 0.40, 0.10],
])


class StochasticBlockModel():

    # fixing density at 0.5
    def __init__(self, num_nodes, num_classes,  homophily, feature_dim, noise, density = 0.5):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.homophily = homophily
        self.feature_dim = feature_dim
        self.density = density
        self.noise = noise
        self.class_means = torch.randn(num_classes, feature_dim)
        # treating the fraction of same class pairs as 1 instead of 1/c (shouldn't matter)

        self.p_in = homophily * density
        # solved out based on the equation density ≈ (1/c)·p_in + (1 - 1/c)·p_out
        self.p_out =   (1 - homophily) * density / (num_classes - 1) if num_classes > 1 else 0

        self.nodes_per_class = num_nodes // num_classes

    def generate(self, num_graphs):
        graphs = []
        
        for graph_id in range(num_graphs):
            node_labels = torch.zeros(self.num_nodes, dtype=torch.long)
            graph_label = graph_id % self.num_classes
            proportions = composition[graph_label]

            counts = (proportions * self.num_nodes).round().long()
            # fix the rounding (trust up to k-1)
            counts[-1] = self.num_nodes - counts[:-1].sum()

            # assign node class labels
            node_labels = torch.cat([torch.full(counts[c], c) for c in range(self.num_classes)])
            
            x = torch.zeros(self.num_nodes, self.feature_dim)
            for c in range(self.num_classes):
                mask = (x == c)
                noise =  torch.randn(mask.sum(), self.feature_dim) * self.noise
                x[mask] =  self.class_means[c] + noise
                

            edges = []

            for u in range(self.num_nodes):
                for v in range(u + 1, self.num_nodes):
                    if (node_labels[u] == node_labels[v]):
                        prob = self.p_in
                    else:
                        prob = self.p_out
                    if (np.random.rand() < prob):
                        edges.append([u, v])
                        edges.append([v, u])

            if len(edges) == 0:
                edges = [[0, 0]]

            edge_index = torch.tensor(edges, dtype=torch.long).T


            graphs.append(Data(x=x, y = torch.tensor([graph_label]), edge_index =edge_index, node_labels=node_labels))

        return graphs



