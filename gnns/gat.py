from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT():
    def __init__(self, in_dim, hidden_layers_dim, num_classes, num_hidden_layers):
        super().__init__()
        self.conv_layers = ModuleList()
        self.conv_layers.append(GATConv(in_channels=in_dim, out_channels=hidden_layers_dim))
        for _ in range(num_hidden_layers - 1):
            self.conv_layers.append(GATConv(in_dim=hidden_layers_dim, out_channels=hidden_layers_dim))
        self.classifier = Linear(in_dim=hidden_layers_dim, out_features=num_classes)

    def forwrd(self, x, edge_index, batch):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        # does the batch thing and returns one vector per graph
        graph_representation = global_mean_pool(x, batch)

        x = self.classifier(graph_representation)

        return x, graph_representation
        