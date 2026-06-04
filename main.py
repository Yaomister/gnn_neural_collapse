from torch_geometric.datasets import TUDataset
from experiments.experiment_real_datasets import run as run_1
from experiments.experiment_sbm import run as run_2


if __name__ == "__main__":
    mutag = TUDataset(root="data/", name= "MUTAG")
    proteins = TUDataset(root="data/", name = "PROTEINS")
    enzymes = TUDataset(root="data/", name = "ENZYMES")

    run_1(dataset=mutag, dataset_name="MUTAG")
    run_1(dataset=proteins, dataset_name="PROTEINS")
    run_1(dataset=enzymes, dataset_name="ENZYMES")

    run_2()