from torch.nn import ModuleList, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCN(Module):
    def __init__(self, in_dim, hidden_layer_dim, num_hidden_layers, num_classes, pool):
        super().__init__()
        self.layers = ModuleList()
        self.layers.append(GCNConv(in_channels=in_dim, out_channels=hidden_layer_dim))
        # add the hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(GCNConv(in_channels=hidden_layer_dim, out_channels=hidden_layer_dim))
        self.pool = pool
        self.classifier = Linear(in_features=hidden_layer_dim, out_features=num_classes)


    def forward(self, x, edge_index, batch):
        intermediate_layers = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            # apply RELU except for the final layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
            intermediate_layers.append(x)

        if self.pool == "mean":
            graph_representation = global_mean_pool(x, batch)
        elif self.pool == "max":
            graph_representation = global_max_pool(x, batch)
        else:
            raise ValueError(f"unknown pool {self.pool}")
            
        x = self.classifier(graph_representation)
        # return the logits, the node feature vectors before the final classifier, and the intermediate layers
        return x, graph_representation, intermediate_layers