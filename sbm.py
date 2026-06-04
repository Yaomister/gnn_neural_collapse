import torch
import numpy as np
from torch_geometric.data import Data


default_composition = torch.tensor([
    [0.1, 0.1, 0.8],  
    [0.1, 0.8, 0.1],  
    [0.8, 0.1, 0.1],
])


class StochasticBlockModel():

    def __init__(self, num_nodes, num_graph_classes, num_node_classes, homophily, feature_dim, noise, density = 0.5, composition = default_composition):
        self.num_nodes = num_nodes
        self.num_node_classes = num_node_classes
        self.num_graph_classes = num_graph_classes
        self.homophily = homophily
        self.feature_dim = feature_dim
        self.density = density
        self.noise_constant = noise
        self.composition = composition
        self.class_means = torch.randn(num_node_classes, feature_dim)

        self.nodes_per_class = num_nodes // num_node_classes

    def _get_probability(self, graph_class):
        current_composition = self.composition[graph_class]

        sum_p_k = sum([p * p for p in current_composition])

        target_edge_homophily = self.homophily * (1 - sum_p_k) + sum_p_k

        p_in  = (target_edge_homophily * self.density) / sum_p_k

        p_out  = (self.density) *  (1 - target_edge_homophily) / (1 - sum_p_k)

        return p_in, p_out


    def generate(self, num_graphs):
        graphs = []
        
        for graph_id in range(num_graphs):
            node_labels = torch.zeros(self.num_nodes, dtype=torch.long)
            graph_class = graph_id % self.num_graph_classes
            proportions = self.composition[graph_class]

            counts = (proportions * self.num_nodes).round().long()
            # fix the rounding (trust up to k-1)
            counts[-1] = self.num_nodes - counts[:-1].sum()

            # assign node class labels
            node_labels = torch.cat([torch.full((counts[c], ), c) for c in range(self.num_node_classes)])
            
            x = torch.zeros(self.num_nodes, self.feature_dim)
            for c in range(self.num_node_classes):
                mask = (node_labels == c)
                noise =  torch.randn(mask.sum(), self.feature_dim) * self.noise_constant
                x[mask] =  self.class_means[c] + noise
                

            edges = []

            p_in, p_out = self._get_probability(graph_class=graph_class)

            for u in range(self.num_nodes):
                for v in range(u + 1, self.num_nodes):
                    if (node_labels[u] == node_labels[v]):
                        prob = p_in
                    else:
                        prob = p_out
                    if (np.random.rand() < prob):
                        edges.append([u, v])
                        edges.append([v, u])

            if len(edges) == 0:
                edges = [[0, 0]]

            edge_index = torch.tensor(edges, dtype=torch.long).T


            graphs.append(Data(x=x, y = torch.tensor([graph_class]), edge_index =edge_index, node_labels=node_labels))

        return graphs



