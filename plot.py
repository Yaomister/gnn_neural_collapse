import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


os.makedirs("figures", exist_ok=True)

homophily = [0.1, 0.3, 0.5, 0.7, 0.9]
pooling = ["max", "mean", "sum"]
noise = [50, 20, 10]
models = ["GCN", "GAT", "GraphSAGE"]
nc_metrics = ["within_class_variance", "class_mean_angles", "class_mean_norms"]

NC_ROW_LABELS = [
    "NC1: Within-Class Variance",
    "NC2: Class Mean Variance",
    "NC2: Class Mean Norms",
]

def load_results():
    """Load all result files from results/ into a dict keyed by filename stem."""
    results = {}
    for file_name in os.listdir("results"):
        if file_name.endswith(".pt"):
            tag = file_name[:-3]
            results[tag] = torch.load(f"results/{tag}.pt", weights_only=False)
    return results


def find_tpt(accuracy, threshold=0.99, patience=5):
    """Return the first epoch where accuracy stays above threshold for consecutive epochs."""
    streak = 0
    for i, acc in enumerate(accuracy):
        streak = streak + 1 if acc > threshold else 0
        if streak > patience:
            return i - patience + 1
    return None


def add_tpt_line(ax, tpt):
    """Draw a dashed red vertical line at the TPT."""
    if tpt is not None:
        ax.axvline(x=tpt, color="red", linestyle="--", linewidth=1.5, label="TPT")


def group_data(results, tag, key):
    """Collect per-seed metric arrays for a given experiment tag."""
    return [results[f"{tag}_{i}"][key] for i in range(5) if f"{tag}_{i}" in results]


def _mean_std(results, tag, key):
    """Return (mean, std) arrays across seeds, or (None, None) if no seeds found."""
    data = group_data(results, tag, key)
    if not data:
        return None, None
    runs = np.asarray(data)
    return runs.mean(axis=0), runs.std(axis=0, ddof=1)


def _plot_band(ax, x, mean, std, color, label=None):
    """Plot a line with a shaded ±1 std band."""
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)


def _set_nc_row_labels(ax):
    """Set NC1/NC2 metric y-axis labels on the leftmost column."""
    for row, label in enumerate(NC_ROW_LABELS):
        ax[row, 0].set_ylabel(label)


# Plot 1: NC metrics vs training epoch for each GNN architecture (ENZYMES dataset)
def plot_experiment_1(results):
    for h in homophily:
        for n in noise:
            for p in pooling:
                fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")
                fig.subplots_adjust(wspace=0.05, hspace=0.1)
                for m in range(len(nc_metrics)):
                    for k in range(len(models)):
                        tag = f"exp1_{models[k]}_ENZYMES_{p.lower()}"
                        mean, std = _mean_std(results, tag, nc_metrics[m])
                        if mean is None:
                            continue
                        epochs = np.arange(1, len(mean) + 1)
                        _plot_band(ax[m, k], epochs, mean, std, "blue")

                        tpt_mean, _ = _mean_std(results, tag, "training_accuracy")
                        if tpt_mean is not None:
                            add_tpt_line(ax[m, k], find_tpt(tpt_mean))

                        if m == 0:
                            ax[m, k].set_title(models[k])
                        ax[2, k].set_xlabel("Epoch")

                    # x ticks are measurement indices; multiply by 5 to recover actual epoch
                    ax[m, 0].xaxis.set_major_locator(ticker.MultipleLocator(10))
                    ax[m, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * 5)}"))
                    ax[m, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax[m, 0].yaxis.set_minor_locator(ticker.NullLocator())

                ax[0, 2].legend(loc="upper right")
                ax[0, 0].set_ylim(0, 8)
                ax[1, 0].set_ylim(0, 0.6)
                ax[2, 0].set_ylim(0, 0.5)
                _set_nc_row_labels(ax)
                fig.savefig(f"figures/graph_1_h{h}_n{n}_{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)


# Plot 2: final NC metric values vs graph homophily for each GNN architecture
def plot_experiment_2(results):
    colors = {50: "red", 20: "green", 10: "blue"}
    for p in pooling:
        fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
        for m in range(len(nc_metrics)):
            for k in range(len(models)):
                for n in noise:
                    final_metrics, final_stds, tracked_h = [], [], []
                    for h in homophily:
                        tag = f"exp2_{models[k]}_h{h}_n{n}_{p.lower()}"
                        mean, std = _mean_std(results, tag, nc_metrics[m])
                        if mean is None:
                            continue
                        final_metrics.append(mean[-1])
                        final_stds.append(std[-1])
                        tracked_h.append(h)

                    if not tracked_h:
                        continue
                    _plot_band(
                        ax[m, k],
                        np.asarray(tracked_h),
                        np.asarray(final_metrics),
                        np.asarray(final_stds),
                        colors[n],
                        label=f"noise={n}",
                    )

                ax[m, k].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax[m, k].yaxis.set_minor_locator(ticker.NullLocator())
                ax[m, k].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax[2, k].set_xlabel("Graph Homophily")
                if m == 0:
                    ax[m, k].set_title(models[k])

        ax[0, 2].legend(loc="upper right")
        _set_nc_row_labels(ax)
        fig.savefig(f"figures/graph_2_p{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


# Plot 3: within-class and between-class Dirichlet energy vs graph homophily
def plot_experiment_3(results):
    for n in noise:
        fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")
        for i, p in enumerate(pooling):
            for k, m in enumerate(models):
                within_means, between_means, within_stds, between_stds, tracked_h = [], [], [], [], []
                for h in homophily:
                    tag = f"exp2_{m}_h{h}_n{n}_{p.lower()}"
                    data = group_data(results, tag, "dirichlet_energies_at_intermediate_layers")
                    if not data:
                        continue
                    # each seed_run is a list of per-epoch snapshots; each snapshot is
                    # [[within_L0, between_L0], ..., [within_Lk, between_Lk]] across GNN layers.
                    # We want the last epoch's last-layer energies.
                    within_seeds = np.asarray([seed_run[-1][-1][0] for seed_run in data])
                    between_seeds = np.asarray([seed_run[-1][-1][1] for seed_run in data])
                    within_means.append(within_seeds.mean())
                    within_stds.append(within_seeds.std(ddof=1))
                    between_means.append(between_seeds.mean())
                    between_stds.append(between_seeds.std(ddof=1))
                    tracked_h.append(h)

                if i == 0:
                    ax[i, k].set_title(m)
                ax[2, k].set_xlabel("Graph Homophily")
                ax[i, k].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax[i, k].yaxis.set_major_locator(ticker.LogLocator(base=10))
                ax[i, k].yaxis.set_minor_locator(ticker.NullLocator())

                if not tracked_h:
                    continue
                _plot_band(ax[i, k], tracked_h, np.asarray(between_means), np.asarray(between_stds), "blue", label="Between-Class Energy")
                _plot_band(ax[i, k], tracked_h, np.asarray(within_means), np.asarray(within_stds), "green", label="Within-Class Energy")

            ax[i, 0].yaxis.set_major_locator(ticker.MultipleLocator(10))
            ax[i, 0].yaxis.set_minor_locator(ticker.NullLocator())

        ax[0, 2].legend(loc="upper right")
        ax[0, 0].set_ylabel("Max-Pool")
        ax[1, 0].set_ylabel("Mean-Pool")
        ax[2, 0].set_ylabel("Sum-Pool")
        fig.savefig(f"figures/graph_3_n{n}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


# Plot 4: NC metrics vs epoch (post-TPT only) for each GNN architecture, one line per pooling operator
def plot_experiment_4(results):
    colors = {"max": "red", "sum": "green", "mean": "blue"}
    for n in noise:
        for h in homophily:
            fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")
            for i, metric in enumerate(nc_metrics):
                for j, m in enumerate(models):
                    for p in pooling:
                        tag = f"exp2_{m}_h{h}_n{n}_{p.lower()}"
                        mean, std = _mean_std(results, tag, metric)
                        tpt_mean, _ = _mean_std(results, tag, "training_accuracy")
                        if mean is None or tpt_mean is None:
                            continue
                        tpt = find_tpt(tpt_mean)
                        if tpt is None:
                            continue

                        epochs = np.arange(tpt, len(mean))
                        _plot_band(ax[i, j], epochs, mean[tpt:], std[tpt:], colors[p], label=f"{p} pool")
                        if i == 0:
                            ax[i, j].set_title(m)
            
                ax[i, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax[i, 0].yaxis.set_minor_locator(ticker.NullLocator())
                # x ticks are measurement indices; multiply by 5 to recover actual epoch
                ax[i, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * 5)}"))

            _set_nc_row_labels(ax)
            ax[0, 2].legend(loc="upper right")
            fig.subplots_adjust(hspace=0.1)
            fig.savefig(f"figures/graph_4_n{n}_h{h}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)


if __name__ == "__main__":
    results = load_results()
    plot_experiment_1(results)
    plot_experiment_2(results)
    plot_experiment_3(results)
    plot_experiment_4(results)
