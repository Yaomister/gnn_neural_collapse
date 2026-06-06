from gnns import GCN, GAT, GraphSAGE

# define a lookup table for the models
models_list = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
}