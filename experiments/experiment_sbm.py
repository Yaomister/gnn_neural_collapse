from training import train
from utils import models_list
import matplotlib.pyplot as plt
from sbm import StochasticBlockModel



def run(model_name, homophily, noise, pool, hidden_dim = 64, num_epochs = 500):

    results = {}

    # the embedding dimensions of the SBM generated graphs
    in_dim = 16
    
    sbm = StochasticBlockModel(num_nodes=50, num_graph_classes=3, num_node_classes=3, homophily=homophily, feature_dim=in_dim, noise=noise)
    model = models_list[model_name](
        in_dim = in_dim, 
        hidden_layer_dim = hidden_dim, 
        num_classes = 3, 
        num_hidden_layers = 3,
        pool= pool
    ) 

    graphs = sbm.generate(1000)

    metrics = train(model=model, graphs=graphs, num_classes=3, num_epochs=num_epochs, measure_energy=True, measure_interval=1)

    return metrics





