import os
import torch
from torch_geometric.datasets import TUDataset
from experiments.experiment_real_datasets import run as run_1
from experiments.experiment_sbm import run as run_2

def build_job_list():
    jobs = []


    models = ["GCN", "GAT", "GraphSAGE"]

    for ds_name in ["MUTAG", "PROTEINS", "ENZYMES"]:
        for model_name in models:
            for p in ["mean", "max"]:
                jobs.append({"experiment": 1, "dataset_name": ds_name, "pool": p, "model_name" : model_name})

    homophily = [0.3, 0.6, 0.9]
    noise = [0.3, 0.6, 0.9]
    pools = ["mean", "max"]
    for m in models:
        for h in homophily:
            for n in noise:
                for p in pools:
                    jobs.append({
                        "experiment": 2,
                        "model_name": m, "homophily": h,
                        "noise": n, "pool": p,
                    })
    return jobs


def run_one_job(job):
    os.makedirs("results", exist_ok=True)

    if job["experiment"] == 1:
        ds = TUDataset(root="data/", name=job["dataset_name"])
        result = run_1(dataset=ds, dataset_name=job["dataset_name"], pool=job['pool'], model_name=job["model_name"])
        tag = f"exp1_{job['model_name']}_{job['dataset_name']}_{job['pool']}"
    else:
        result = run_2(
            model_name=job["model_name"],
            homophily=job["homophily"],
            noise=job["noise"],
            pool=job["pool"],
        )
        tag = f"exp2_{job['model_name']}_h{job['homophily']}_n{job['noise']}_{job['pool']}"

    torch.save(result, f"results/{tag}.pt")
    print(f"finished {tag}")


if __name__ == "__main__":
    jobs = build_job_list()
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        for job in jobs:
            run_one_job(job)
    else:
        run_one_job(jobs[int(task_id)])
