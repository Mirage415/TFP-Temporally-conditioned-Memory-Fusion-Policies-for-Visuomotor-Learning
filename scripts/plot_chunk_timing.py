import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro


def _read_rows(csv_path: pathlib.Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "task_id": int(row["task_id"]),
                    "task_description": row["task_description"],
                    "episode_idx": int(row["episode_idx"]),
                    "chunk_idx": int(row["chunk_idx"]),
                    "planned_steps": int(row["planned_steps"]),
                    "executed_steps": int(row["executed_steps"]),
                    "infer_sec": float(row["infer_sec"]),
                    "exec_sec": float(row["exec_sec"]),
                    "episode_step_start": int(row["episode_step_start"]),
                    "episode_step_end_exclusive": int(row["episode_step_end_exclusive"]),
                    "completed_reason": row["completed_reason"],
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _build_output_path(csv_path: pathlib.Path) -> pathlib.Path:
    return csv_path.with_name(f"{csv_path.stem}_exec_sec_by_episode.png")


def main(
    csv_path: str,
    output_path: str | None = None,
    metric: str = "exec_sec",
) -> None:
    csv_file = pathlib.Path(csv_path)
    rows = _read_rows(csv_file)
    metric_name = metric
    if metric_name not in {"exec_sec", "infer_sec"}:
        raise ValueError(f"Unsupported metric: {metric_name}")

    output_file = pathlib.Path(output_path) if output_path else _build_output_path(csv_file)
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["task_id"], row["episode_idx"]), []).append(row)

    keys = sorted(grouped)
    num_panels = len(keys)
    num_cols = min(3, num_panels)
    num_rows = math.ceil(num_panels / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.2 * num_cols, 3.8 * num_rows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, key in zip(flat_axes, keys, strict=False):
        episode_rows = sorted(grouped[key], key=lambda row: row["chunk_idx"])
        x = [row["chunk_idx"] for row in episode_rows]
        y = [row[metric_name] for row in episode_rows]
        ax.plot(x, y, color="#1f77b4", linewidth=1.6, marker="o", markersize=3.5)

        partial_rows = [row for row in episode_rows if row["executed_steps"] < row["planned_steps"]]
        if partial_rows:
            ax.scatter(
                [row["chunk_idx"] for row in partial_rows],
                [row[metric_name] for row in partial_rows],
                color="#d62728",
                s=28,
                zorder=3,
                label="partial chunk",
            )

        task_id, episode_idx = key
        final_row = episode_rows[-1]
        ax.set_title(
            f"task={task_id} episode={episode_idx} chunks={len(episode_rows)} end={final_row['completed_reason']}",
            fontsize=10,
        )
        ax.set_xlabel("chunk_idx")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

        desc = episode_rows[0]["task_description"]
        wrapped_desc = "\n".join([desc[i : i + 55] for i in range(0, len(desc), 55)])
        ax.text(
            0.01,
            0.99,
            wrapped_desc,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
        if partial_rows:
            ax.legend(loc="lower right", fontsize=8)

    for ax in flat_axes[num_panels:]:
        ax.axis("off")

    fig.suptitle(f"{csv_file.name}: {metric_name} by chunk", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    print(output_file)


if __name__ == "__main__":
    tyro.cli(main)
