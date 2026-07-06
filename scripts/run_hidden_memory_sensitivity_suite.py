import argparse
import csv
import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as _config
import openpi.training.temporal_memory_loader as _temporal_memory_loader

try:
    from scripts import run_hidden_memory_intervention as _intervention
except ModuleNotFoundError:
    import run_hidden_memory_intervention as _intervention


def _metrics(action: np.ndarray, ref_action: np.ndarray, *, dims: int | None = None) -> dict[str, float]:
    if dims is not None:
        action = action[..., :dims]
        ref_action = ref_action[..., :dims]
    delta = np.asarray(action - ref_action, dtype=np.float32)
    flat = delta.reshape(-1)
    action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
    ref_flat = np.asarray(ref_action, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(action_flat) * np.linalg.norm(ref_flat)
    cosine = float(np.dot(action_flat, ref_flat) / denom) if denom > 0 else float("nan")
    return {
        "mean_l2_per_step": float(np.linalg.norm(delta, axis=-1).mean()),
        "first_action_l2": float(np.linalg.norm(delta[0])),
        "max_abs_delta": float(np.max(np.abs(flat))),
        "mean_abs_delta": float(np.mean(np.abs(flat))),
        "cosine": cosine,
    }


def _load_action_stats(path: pathlib.Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    stats = data.get("norm_stats", data).get("actions")
    if not stats:
        return None
    return {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}


def _unnormalize_actions(actions: np.ndarray, stats: dict[str, np.ndarray] | None) -> np.ndarray:
    if stats is None:
        return actions
    q01 = stats.get("q01")
    q99 = stats.get("q99")
    if q01 is not None and q99 is not None:
        dim = q01.shape[-1]
        out = actions.copy()
        out[..., :dim] = (out[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return out
    mean = stats["mean"]
    std = stats["std"]
    dim = mean.shape[-1]
    out = actions.copy()
    out[..., :dim] = out[..., :dim] * (std + 1e-6) + mean
    return out


def _select_targets(dataset, target_local_chunk_index: int, num_targets: int, stride: int) -> list[int]:
    selected = []
    seen_episodes = set()
    for sample_index in range(0, len(dataset), max(stride, 1)):
        sample = dataset[sample_index]
        if target_local_chunk_index >= sample["chunks"]["chunk_ids"].shape[0]:
            continue
        chunk_id = int(np.asarray(sample["chunks"]["chunk_ids"][target_local_chunk_index]).item())
        valid = bool(np.asarray(sample["chunks"]["mask"][target_local_chunk_index]).item())
        if chunk_id < 0 or not valid:
            continue
        episode_id = _intervention._sample_episode_id(dataset, sample_index)
        if episode_id in seen_episodes:
            continue
        selected.append(sample_index)
        seen_episodes.add(episode_id)
        if len(selected) >= num_targets:
            break
    if not selected:
        raise ValueError("Could not find any valid target samples.")
    return selected


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Batch hidden-memory sensitivity and noise baseline suite.")
    parser.add_argument("--config-name", default="tfp_a1")
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--temporal-cache-dir", default="assets/tfp_a1/temporal_cache", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--norm-stats-path", type=pathlib.Path, default=None)
    parser.add_argument("--target-local-chunk-index", type=int, default=2)
    parser.add_argument("--target-sample-indices", default="")
    parser.add_argument("--num-targets", type=int, default=8)
    parser.add_argument("--target-stride", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-noise-samples", type=int, default=8)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, temporal_cache_dir=str(args.temporal_cache_dir))
    dataset = _temporal_memory_loader.CachedTemporalMemoryDataset(
        args.temporal_cache_dir,
        config.model.tbptt_num_chunks,
        config.model.tbptt_chunk_len,
    )
    if args.norm_stats_path is None:
        args.norm_stats_path = (
            pathlib.Path(config.assets_dirs) / config.data.asset_id / "norm_stats.json"
        )
    action_stats = _load_action_stats(args.norm_stats_path)

    if args.target_sample_indices:
        target_sample_indices = [int(value) for value in args.target_sample_indices.split(",") if value.strip()]
    else:
        target_sample_indices = _select_targets(
            dataset,
            args.target_local_chunk_index,
            args.num_targets,
            args.target_stride,
        )

    _intervention._maybe_install_checkpoint_compat(args.checkpoint_dir)
    model = _intervention._load_model(config, args.checkpoint_dir)

    rows = []
    per_dim_rows = []
    for target_ordinal, target_sample_index in enumerate(target_sample_indices):
        target_sample = _intervention._to_jax(dataset[target_sample_index])
        target_chunk_id = int(
            np.asarray(target_sample["chunks"]["chunk_ids"][args.target_local_chunk_index]).item()
        )
        target_episode_id = _intervention._sample_episode_id(dataset, target_sample_index)
        hidden_by_chunk, observation_by_chunk, _ = _intervention._compute_episode_rollout(
            model,
            dataset,
            target_episode_id,
            stop_after_chunk_id=target_chunk_id,
        )
        observation = observation_by_chunk[target_chunk_id]

        episode_ids = sorted({int(window.episode_index) for window in getattr(dataset, "_windows")})
        other_episode_id = next((episode_id for episode_id in episode_ids if episode_id != target_episode_id), None)
        other_hidden_by_chunk = {}
        if other_episode_id is not None:
            other_hidden_by_chunk, _, _ = _intervention._compute_episode_rollout(
                model,
                dataset,
                other_episode_id,
                stop_after_chunk_id=target_chunk_id,
            )

        variants = _intervention._build_variants(
            hidden_by_chunk,
            other_hidden_by_chunk,
            target_chunk_id,
            args.seed + 100 + target_ordinal,
        )
        shared_noise = jax.random.normal(
            jax.random.key(args.seed + target_ordinal),
            (1, model.action_horizon, model.action_dim),
            dtype=observation.state.dtype,
        )

        actions = {}
        for name, hidden in variants.items():
            hidden = hidden[None, :]
            memory_cond = model._temporal_condition_from_hidden(hidden, jnp.ones((1,), dtype=jnp.bool_))
            action = model._sample_actions_with_memory(
                jax.random.key(args.seed + 1 + target_ordinal),
                observation,
                num_steps=args.num_steps,
                noise=shared_noise,
                memory_cond=memory_cond,
            )
            actions[name] = np.asarray(action[0], dtype=np.float32)

        ref = actions["target_normal"]
        unnorm_ref = _unnormalize_actions(ref[None], action_stats)[0]
        for variant, action in actions.items():
            unnorm_action = _unnormalize_actions(action[None], action_stats)[0]
            row = {
                "kind": "memory",
                "target_ordinal": target_ordinal,
                "target_sample_index": target_sample_index,
                "target_episode_id": target_episode_id,
                "target_local_chunk_index": args.target_local_chunk_index,
                "target_chunk_id": target_chunk_id,
                "other_episode_id": -1 if other_episode_id is None else other_episode_id,
                "variant": variant,
                "reference": "target_normal",
            }
            row.update(_prefixed("norm_all32", _metrics(action, ref)))
            row.update(_prefixed("norm_first7", _metrics(action, ref, dims=7)))
            row.update(_prefixed("unnorm_first7", _metrics(unnorm_action, unnorm_ref, dims=7)))
            rows.append(row)

            delta = np.abs(action[..., :7] - ref[..., :7]).mean(axis=0)
            unnorm_delta = np.abs(unnorm_action[..., :7] - unnorm_ref[..., :7]).mean(axis=0)
            for dim in range(7):
                per_dim_rows.append(
                    {
                        "kind": "memory",
                        "target_ordinal": target_ordinal,
                        "target_sample_index": target_sample_index,
                        "variant": variant,
                        "dim": dim,
                        "norm_mean_abs_delta": float(delta[dim]),
                        "unnorm_mean_abs_delta": float(unnorm_delta[dim]),
                    }
                )

        normal_hidden = hidden_by_chunk[target_chunk_id][None, :]
        memory_cond = model._temporal_condition_from_hidden(normal_hidden, jnp.ones((1,), dtype=jnp.bool_))
        noise_actions = []
        noise_labels = []
        for i in range(args.num_noise_samples):
            noise_seed = args.seed + 10_000 * (target_ordinal + 1) + i
            noise = jax.random.normal(
                jax.random.key(noise_seed),
                (1, model.action_horizon, model.action_dim),
                dtype=observation.state.dtype,
            )
            action = model._sample_actions_with_memory(
                jax.random.key(noise_seed + 123),
                observation,
                num_steps=args.num_steps,
                noise=noise,
                memory_cond=memory_cond,
            )
            noise_actions.append(np.asarray(action[0], dtype=np.float32))
            noise_labels.append(f"noise_seed_{noise_seed}")

        noise_ref = noise_actions[0]
        unnorm_noise_ref = _unnormalize_actions(noise_ref[None], action_stats)[0]
        for label, action in zip(noise_labels, noise_actions, strict=True):
            unnorm_action = _unnormalize_actions(action[None], action_stats)[0]
            row = {
                "kind": "noise",
                "target_ordinal": target_ordinal,
                "target_sample_index": target_sample_index,
                "target_episode_id": target_episode_id,
                "target_local_chunk_index": args.target_local_chunk_index,
                "target_chunk_id": target_chunk_id,
                "other_episode_id": -1,
                "variant": label,
                "reference": noise_labels[0],
            }
            row.update(_prefixed("norm_all32", _metrics(action, noise_ref)))
            row.update(_prefixed("norm_first7", _metrics(action, noise_ref, dims=7)))
            row.update(_prefixed("unnorm_first7", _metrics(unnorm_action, unnorm_noise_ref, dims=7)))
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "suite_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_dim_path = args.output_dir / "suite_per_dim.csv"
    with per_dim_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_dim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_dim_rows)

    run_config_path = args.output_dir / "suite_run_config.json"
    run_config_path.write_text(json.dumps({**vars(args), "target_sample_indices": target_sample_indices}, indent=2, default=str))

    print(f"Wrote {metrics_path}")
    print(f"Wrote {per_dim_path}")
    print(f"Wrote {run_config_path}")
    print(f"target_sample_indices={target_sample_indices}")


if __name__ == "__main__":
    main()
