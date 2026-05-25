from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool


class GCN_Max_Pool():
    def __init__(self, in_dim, num_classes, hidden_layer_dim, num_hidden_layers):
        super().__init__()
        self.layers = ModuleList()
        self.layers.append(GCNConv(in_channels=in_dim, out_channels=hidden_layer_dim))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(GCNConv(in_channels=hidden_layer_dim, out_channels=hidden_layer_dim))
        self.classifier = Linear(in_features=hidden_layer_dim, out_features=num_classes)
            
    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        graph_representation = global_max_pool(x)

        x = self.classifier(graph_representation)
        
        return x, graph_representation

        