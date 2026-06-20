from torch.nn import ModuleList, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool


class GAT(Module):
    """Graph Attention Network for graph classification.

    Stacks num_hidden_layers GATConv layers with multi-head attention (num_heads heads).
    hidden_layer_dim must be divisible by num_heads since each head outputs hidden_layer_dim //
    num_heads features and they are concatenated back. Follows the same pooling and classification
    structure as GCN. Returns (logits, graph_repr, intermediate_node_features).
    """

    def __init__(self, in_dim, hidden_layer_dim, num_classes, num_hidden_layers, pool, num_heads=4):
        super().__init__()
        # make sure the hidden layer feature embedding dimensions can be evenly split upon the heads
        assert hidden_layer_dim % num_heads == 0
        self.layers = ModuleList()
        self.layers.append(GATConv(in_channels=in_dim, out_channels=hidden_layer_dim // num_heads, heads=num_heads))
        # add the hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(GATConv(in_channels=hidden_layer_dim, out_channels=hidden_layer_dim // num_heads, heads=num_heads))
        self.classifier = Linear(in_features=hidden_layer_dim, out_features=num_classes)
        self.pool = pool

    def forward(self, x, edge_index, batch):
        intermediate_layers = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            # run RELU except for the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
            intermediate_layers.append(x)
        # global pooling
        if self.pool == "mean":
            graph_representation = global_mean_pool(x, batch)
        elif self.pool == "max":
            graph_representation = global_max_pool(x, batch)
        elif self.pool == "sum":
            graph_representation = global_add_pool(x, batch)
        else:
            raise ValueError(f"unknown pool {self.pool}")
        x = self.classifier(graph_representation)
        # return the logits, the node feature vectors before the final classifier, and all intermediate layers
        return x, graph_representation, intermediate_layers