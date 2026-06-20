import torch
import numpy as np
from torch_geometric.data import Data


# the node composition of a graph to make graph classification possible
default_composition = torch.tensor([
    [0.1, 0.1, 0.8],  
    [0.1, 0.8, 0.1],  
    [0.8, 0.1, 0.1],
])

class StochasticBlockModel():
    """Generates synthetic graph-classification datasets with controlled homophily.

    Each graph belongs to one of num_graph_classes classes, determined by the node-class
    composition (proportion of each node type). Edges are drawn with probability p_in for
    same-class node pairs and p_out for cross-class pairs, where p_in and p_out are calibrated
    so that the expected adjusted edge homophily matches the target homophily value.
    """

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
        """Compute within-class (p_in) and between-class (p_out) edge probabilities.

        Solves backwards from the target adjusted homophily to per-class edge probabilities,
        accounting for node-class imbalance via the class proportion vector S = sum(p_k^2).
        """
        current_composition = self.composition[graph_class]

        sum_p_k = sum([p * p for p in current_composition])

        # using adjusted homophily because we have node-level class imbalance
        # this is solving backwards from adjusted homophily to the specific edge homophily for each class
        target_edge_homophily = self.homophily * (1 - sum_p_k) + sum_p_k

        # from h_edge ​ =h_adj​(1−S) + S, S is the expected fraction of edges within the same class, so 1 - S is the expected fraction of edges of different classes
        p_in  = (target_edge_homophily * self.density) / sum_p_k

        p_out  = (self.density) *  (1 - target_edge_homophily) / (1 - sum_p_k)

        return p_in, p_out


    def generate(self, num_graphs):
        """Generate num_graphs PyG Data objects with node features and graph-level labels."""
        graphs = []
        
        # generate graphs
        for graph_id in range(num_graphs):
            # make a matrix where each column represents a node
            node_labels = torch.zeros(self.num_nodes, dtype=torch.long)
            # get the class which the graph belongs to 
            graph_class = graph_id % self.num_graph_classes
            # get the proportion of node classes based on the class the graph belongs to 
            proportions = self.composition[graph_class]

            counts = (proportions * self.num_nodes).round().long()
            # fix the rounding (trust up to k-1)
            counts[-1] = self.num_nodes - counts[:-1].sum()
            # assign node class labels
            node_labels = torch.cat([torch.full((counts[c], ), c) for c in range(self.num_node_classes)])
            
            # the feature vectors, starting off as all 0
            x = torch.zeros(self.num_nodes, self.feature_dim)

            for c in range(self.num_node_classes):
                # for nodes of a class, center it around the same class mean, but add noise to it
                mask = (node_labels == c)
                noise =  torch.randn(mask.sum(), self.feature_dim) * self.noise_constant
                x[mask] =  self.class_means[c] + noise
                
            edges = []

            p_in, p_out = self._get_probability(graph_class=graph_class)

            # add edges based probabilities of between-class and within-class edges
            for u in range(self.num_nodes):
                for v in range(u + 1, self.num_nodes):
                    if (node_labels[u] == node_labels[v]):
                        prob = p_in
                    else:
                        prob = p_out
                    if (np.random.rand() < prob):
                        edges.append([u, v])
                        edges.append([v, u])

            # in case there's no edges, add a self loop
            if len(edges) == 0:
                edges = [[0, 0]]

            # (2, N), 0-th row represent the starting point of edges, and the 1st row represents the ending point of edges
            edge_index = torch.tensor(edges, dtype=torch.long).T

            # add it to a PyTorch dataset for training
            graphs.append(Data(x=x, y = torch.tensor([graph_class]), edge_index =edge_index, node_labels=node_labels))
            
        return graphs



