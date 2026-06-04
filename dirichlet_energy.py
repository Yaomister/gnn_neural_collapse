import torch

def calculate_dirichlet_energy(graph_at_layer, edge_index, node_labels ):
    # graph_at_layer: (number of nodes, embedding dimension)
    # edge_index: (2, number of edges for that graph)
    # node_labels: (number of nodes)

    within_class_energy = 0.0
    between_class_energy = 0.0
    
    num_nodes = graph_at_layer.size(0)

    degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()

    for i in range(len(edge_index[0])):
        source = edge_index[0][i]
        to = edge_index[1][i]
        d_source = degrees[source]
        d_to = degrees[to]
        energy_vector = (graph_at_layer[source]/torch.sqrt(1 + d_source)) - (graph_at_layer[to]/torch.sqrt(1 + d_to))
        energy = (energy_vector ** 2).sum()
        if node_labels[source] == node_labels[to]:
            within_class_energy += energy 
        else:
            between_class_energy += energy


    
    within_class_energy = within_class_energy * 1/2
    between_class_energy = between_class_energy * 1/2

    return within_class_energy, between_class_energy
