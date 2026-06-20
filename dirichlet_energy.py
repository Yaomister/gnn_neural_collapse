import torch


def calculate_dirichlet_energy(graph_at_layer, edge_index, node_labels):
    """Compute within-class and between-class Dirichlet energy for a graph's node features.

    Split by whether edge (u,v) connects nodes of the same class (within) or different classes (between).
    Features are L2-normalized before computing so scale doesn't affect the result.

    Returns (within_energy, between_energy) as scalar tensors.
    """
    # graph_at_layer: (number of nodes, embedding dimension)
    # edge_index: (2, number of edges for that graph)
    # node_labels: (number of nodes)

    # normalize the node feature vectors so the values are not affected by the norms
    graph_at_layer = graph_at_layer / (graph_at_layer.norm(dim=1, keepdim=True) + 1e-8)

    # the edge index's 0th row represent the starting point of edges, and the 1st row represent the end point of edges
    num_nodes = graph_at_layer.size(0)
    degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
    edge_start, edge_end = edge_index[0], edge_index[1]

    # following the fomula for the dirichlet energy
    scale = 1.0 / torch.sqrt(1.0 + degrees)
    # make scaled have dimension (N, 1), so it can be broadcasted against graph_at_layer, which has dimension (N, D)
    scaled = graph_at_layer * scale.unsqueeze(1)


    diff = scaled[edge_start] - scaled[edge_end]
    # element wise squaring, then you collapse the squared feature vector difference into a single number for each node
    per_edge = (diff * diff).sum(dim = 1)

    # split it into within-class and between-class dirichlet energy
    same = node_labels[edge_start] == node_labels[edge_end]

    within = 0.5 * per_edge[same].sum()
    between = 0.5 * per_edge[~same].sum()

    # return the energies
    return within, between



