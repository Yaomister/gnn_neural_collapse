import torch


def calculate_dirichlet_energy(graph_at_layer, edge_index, node_labels ):
    # graph_at_layer: (number of nodes, embedding dimension)
    # edge_index: (2, number of edges for that graph)
    # node_labels: (number of nodes)
    
    num_nodes = graph_at_layer.size(0)
    degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
    edge_start, edge_end = edge_index[0], edge_index[1]
    
    scale = 1.0 / torch.sqrt(1.0 + degrees)
    scaled = graph_at_layer * scale.unsqueeze(1)

    diff = scaled[edge_start] - scaled[edge_end]
    per_edge = (diff * diff).sum(dim = 1)

    same = node_labels[edge_start] == node_labels[edge_end]
    within = 0.5 * per_edge[same].sum()
    between = 0.5 * per_edge[~same].sum()


    return within, between



