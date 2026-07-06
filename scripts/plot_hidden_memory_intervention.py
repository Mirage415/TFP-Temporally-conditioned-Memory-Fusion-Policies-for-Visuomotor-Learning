import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Plot hidden-memory intervention outputs.")
    parser.add_argument(
        "--experiment-dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/hidden_memory_intervention_a1_9999_priority"),
    )
    parser.add_argument("--reference", default="target_normal")
    args = parser.parse_args()

    metrics_path = args.experiment_dir / "metrics.csv"
    results_path = args.experiment_dir / "results.npz"
    if not metrics_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Expected {metrics_path} and {results_path}")

    rows = _load_metrics(metrics_path)
    data = np.load(results_path)
    variant_names = [str(v) for v in data["variant_names"]]
    actions = data["actions"]
    ref_index = variant_names.index(args.reference)
    ref = actions[ref_index]
    deltas = actions - ref[None, ...]

    plot_dir = args.experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["variant"] for row in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "mean_l2_per_step", "Mean L2 per action step"),
        (axes[1], "first_action_l2", "First action L2"),
        (axes[2], "max_abs_delta", "Max absolute action delta"),
    ]:
        ax.bar(x, [float(row[key]) for row in rows], color="#4c78a8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(plot_dir / "metric_bars.png", dpi=180)
    plt.close(fig)

    heatmap = np.linalg.norm(deltas, axis=-1)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    im = ax.imshow(heatmap, aspect="auto", cmap="magma")
    ax.set_title("Action delta L2 by variant and horizon step")
    ax.set_xlabel("Action horizon step")
    ax.set_ylabel("Hidden variant")
    ax.set_yticks(np.arange(len(variant_names)))
    ax.set_yticklabels(variant_names)
    ax.set_xticks(np.arange(actions.shape[1]))
    fig.colorbar(im, ax=ax, label="L2(action - reference)")
    fig.savefig(plot_dir / "action_delta_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for idx, name in enumerate(variant_names):
        if name == args.reference:
            continue
        ax.plot(np.arange(actions.shape[1]), heatmap[idx], marker="o", linewidth=1.5, label=name)
    ax.set_title("Action delta L2 across horizon")
    ax.set_xlabel("Action horizon step")
    ax.set_ylabel("L2(action - reference)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(plot_dir / "action_delta_lines.png", dpi=180)
    plt.close(fig)

    for idx, name in enumerate(variant_names):
        if name == args.reference:
            continue
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        vmax = float(np.max(np.abs(deltas[idx])))
        vmax = max(vmax, 1e-6)
        im = ax.imshow(deltas[idx].T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Per-dimension action delta: {name}")
        ax.set_xlabel("Action horizon step")
        ax.set_ylabel("Action dim")
        fig.colorbar(im, ax=ax, label="action - reference")
        fig.savefig(plot_dir / f"delta_dims_{_sanitize(name)}.png", dpi=180)
        plt.close(fig)

    print(f"Wrote plots to {plot_dir}")
    for path in sorted(plot_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
