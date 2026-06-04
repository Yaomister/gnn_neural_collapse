from torch.nn import ModuleList, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(Module):
    def __init__(self, in_dim, num_classes, num_hidden_layers, hidden_layer_dim):
        super().__init__()
        self.conv_layers = ModuleList()
        self.conv_layers.append(SAGEConv(in_channels=in_dim, out_channels=hidden_layer_dim))
        
        for _ in range(num_hidden_layers - 1):
            self.conv_layers.append(SAGEConv(in_channels=hidden_layer_dim, out_channels=hidden_layer_dim))

        self.classifier = Linear(in_dim=hidden_layer_dim, out_features=num_classes)

    def forward(self, x, batch, edge_index):
        intermediate_layers = []
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            intermediate_layers.append(x)

        graph_representation = global_mean_pool(x, batch)

        x = self.classifier(graph_representation)

        return x, graph_representation, intermediate_layers

            
