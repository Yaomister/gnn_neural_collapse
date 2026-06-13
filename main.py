import os
import torch
import random
import numpy as np
from torch_geometric.datasets import TUDataset
from experiments.experiment_sbm import run as run_2
from experiments.experiment_real_datasets import run as run_1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_job_list(seeds=[0, 1]):
    jobs = []
    for s in seeds:
        # the models we're training
        models = ["GCN", "GAT", "GraphSAGE"]

        for ds_name in ["ENZYMES"]:
            for model_name in models:
                for p in ["sum"]:
                    jobs.append({"experiment": 1, "dataset_name": ds_name, "pool": p, "model_name" : model_name, "seed": s})

        # the homophilies
        homophily = [0.1, 0.3, 0.5, 0.7, 0.9]
        # the noise
        noise = [10, 20, 50]
        # the global pooling step
        pools = ["sum"]
        for m in models:
            for h in homophily:
                for n in noise:
                    for p in pools:
                        jobs.append({
                            "experiment": 2,
                            "model_name": m, "homophily": h,
                            "noise": n, "pool": p,
                            "seed":s
                        })
    return jobs

def run_one_job(job):
    os.makedirs("results", exist_ok=True)
    set_seed(job["seed"])

    if job["experiment"] == 1:
        ds = TUDataset(root="data/", name=job["dataset_name"])
        tag = f"exp1_{job['model_name']}_{job['dataset_name']}_{job['pool']}_{job['seed']}"
        print(f"running {tag}")
        result = run_1(dataset=ds, dataset_name=job["dataset_name"], pool=job['pool'], model_name=job["model_name"])
    else:
        result = run_2(
            model_name=job["model_name"],
            homophily=job["homophily"],
            noise=job["noise"],
            pool=job["pool"],
        )
        tag = f"exp2_{job['model_name']}_h{job['homophily']}_n{job['noise']}_{job['pool']}_{job['seed']}"

    # save the results, does override
    torch.save(result, f"results/{tag}.pt")


if __name__ == "__main__":
    jobs = build_job_list()
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    # run all the jobs
    if task_id is None:
        for job in jobs:
            run_one_job(job)
    else:
        run_one_job(jobs[int(task_id)])
