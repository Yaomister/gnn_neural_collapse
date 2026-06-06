from training import train
from utils import models_list
from sbm import StochasticBlockModel



# Experiment 2, 3, and 4: show that pooling and homophily affect the speed and completeness of neural collapse
def run(model_name, homophily, noise, pool, hidden_dim = 64, num_epochs = 200):

    # the embedding dimensions of the SBM generated graphs
    in_dim = 16
    
    # the SBM
    sbm = StochasticBlockModel(num_nodes=50, num_graph_classes=3, num_node_classes=3, homophily=homophily, feature_dim=in_dim, noise=noise)

    # load in the models
    model = models_list[model_name](
        in_dim = in_dim, 
        hidden_layer_dim = hidden_dim, 
        num_classes = 3, 
        num_hidden_layers = 3,
        pool= pool
    ) 

    # generate the graphs
    graphs = sbm.generate(1000)

    # train the models till TPT
    metrics = train(model=model, graphs=graphs, num_classes=3, num_epochs=num_epochs, measure_energy=True, measure_interval=5)

    return metrics





