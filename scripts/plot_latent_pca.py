import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro


def _load_latent_dir(latent_dir: pathlib.Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    episode_files = sorted(latent_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {latent_dir}")

    hidden_rows: list[np.ndarray] = []
    metadata: dict[str, list[np.ndarray]] = {
        "task_id": [],
        "episode_idx": [],
        "chunk_idx": [],
        "env_step": [],
        "progress": [],
    }
    task_descriptions: list[str] = []

    for episode_file in episode_files:
        data = np.load(episode_file, allow_pickle=False)
        hidden_rows.append(np.asarray(data["hidden"], dtype=np.float32))
        for key in metadata:
            metadata[key].append(np.asarray(data[key]))
        task_description = data["task_description"]
        if np.asarray(task_description).ndim == 0:
            task_descriptions.extend([str(task_description.item())] * len(data["hidden"]))
        else:
            task_descriptions.extend([str(x) for x in task_description.tolist()])

    flat_metadata = {key: np.concatenate(values, axis=0) for key, values in metadata.items()}
    flat_metadata["task_description"] = np.asarray(task_descriptions)
    return np.concatenate(hidden_rows, axis=0), flat_metadata


def _sample_rows(
    hidden: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if hidden.shape[0] <= max_points:
        return hidden, metadata
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(hidden.shape[0], size=max_points, replace=False))
    return hidden[indices], {key: value[indices] for key, value in metadata.items()}


def _compute_pca(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(hidden, dtype=np.float32)
    mean = np.mean(hidden, axis=0, keepdims=True)
    std = np.std(hidden, axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    standardized = (hidden - mean) / std

    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:2]
    projected = standardized @ components.T

    explained_variance = (singular_values**2) / max(standardized.shape[0] - 1, 1)
    explained_ratio = explained_variance[:2] / max(np.sum(explained_variance), 1e-12)
    return projected, explained_ratio


def _task_labels(task_ids: np.ndarray, task_descriptions: np.ndarray) -> dict[int, str]:
    labels: dict[int, str] = {}
    for task_id in np.unique(task_ids):
        mask = task_ids == task_id
        descriptions = task_descriptions[mask]
        label = str(descriptions[0]) if len(descriptions) else f"task_{int(task_id)}"
        labels[int(task_id)] = label
    return labels


def main(
    latent_dir: str,
    output_path: str | None = None,
    max_points: int = 5000,
    seed: int = 0,
    point_size: float = 8.0,
    alpha: float = 0.65,
) -> None:
    latent_path = pathlib.Path(latent_dir)
    hidden, metadata = _load_latent_dir(latent_path)
    hidden, metadata = _sample_rows(hidden, metadata, max_points=max_points, seed=seed)
    projected, explained_ratio = _compute_pca(hidden)

    output_file = pathlib.Path(output_path) if output_path else latent_path / "latent_pca.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    task_ids = metadata["task_id"].astype(np.int32)
    progress = metadata["progress"].astype(np.float32)
    task_descriptions = metadata["task_description"]
    labels = _task_labels(task_ids, task_descriptions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    unique_task_ids = sorted(labels)
    cmap = plt.get_cmap("tab10", max(len(unique_task_ids), 1))
    for color_idx, task_id in enumerate(unique_task_ids):
        mask = task_ids == task_id
        axes[0].scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=point_size,
            alpha=alpha,
            color=cmap(color_idx),
            label=f"{task_id}: {labels[task_id]}",
        )
    axes[0].set_title("PCA Colored By Task")
    axes[0].set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    axes[0].grid(True, alpha=0.25)
    if len(unique_task_ids) <= 12:
        axes[0].legend(fontsize=8, loc="best")

    progress_plot = axes[1].scatter(
        projected[:, 0],
        projected[:, 1],
        c=progress,
        s=point_size,
        alpha=alpha,
        cmap="viridis",
    )
    axes[1].set_title("PCA Colored By Episode Progress")
    axes[1].set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    axes[1].grid(True, alpha=0.25)
    colorbar = fig.colorbar(progress_plot, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized Progress")

    suite_names = np.unique(
        [
            np.load(path, allow_pickle=False)["task_suite_name"].item()
            for path in sorted(latent_path.glob("episode_*.npz"))[:10]
        ]
    )
    suite_label = ", ".join([name for name in suite_names if name]) or "unknown_suite"
    fig.suptitle(
        f"PCA of Temporal Latents | suite={suite_label} | points={projected.shape[0]} | episodes={len(list(latent_path.glob('episode_*.npz')))}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")

    projected_output = output_file.with_suffix(".npz")
    np.savez_compressed(
        projected_output,
        projected=projected.astype(np.float32),
        explained_ratio=explained_ratio.astype(np.float32),
        task_id=task_ids,
        progress=progress,
        task_description=task_descriptions,
    )
    print(output_file)
    print(projected_output)


if __name__ == "__main__":
    tyro.cli(main)
