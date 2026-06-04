from torch.nn import ModuleList, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(Module):
    def __init__(self, in_dim, hidden_layers_dim, num_classes, num_hidden_layers, num_heads = 4):
        super().__init__()
        self.layers = ModuleList()
        self.layers.append(GATConv(in_channels=in_dim, out_channels=hidden_layers_dim // num_heads, heads=num_heads))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(GATConv(in_dim=hidden_layers_dim, out_channels=hidden_layers_dim // num_heads, heads=num_heads))
        self.classifier = Linear(in_dim=hidden_layers_dim, out_features=num_classes)

    def forwrd(self, x, edge_index, batch):
        intermediate_layers = []
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            intermediate_layers.append(x)
        # does the batch thing and returns one vector per graph
        graph_representation = global_mean_pool(x, batch)

        x = self.classifier(graph_representation)

        return x, graph_representation, intermediate_layers
        