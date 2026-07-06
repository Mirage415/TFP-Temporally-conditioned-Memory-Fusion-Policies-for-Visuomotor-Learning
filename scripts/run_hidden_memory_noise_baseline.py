import argparse
import csv
import dataclasses
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


def _metrics(action, ref_action):
    delta = np.asarray(action - ref_action, dtype=np.float32)
    action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
    ref_flat = np.asarray(ref_action, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(action_flat) * np.linalg.norm(ref_flat)
    cosine = float(np.dot(action_flat, ref_flat) / denom) if denom > 0 else float("nan")
    return {
        "mean_l2_per_step": float(np.linalg.norm(delta, axis=-1).mean()),
        "first_action_l2": float(np.linalg.norm(delta[0])),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "cosine": cosine,
    }


def main():
    parser = argparse.ArgumentParser(description="Same hidden-memory, different diffusion-noise baseline.")
    parser.add_argument("--config-name", default="tfp_a1")
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--temporal-cache-dir", default="assets/tfp_a1/temporal_cache", type=pathlib.Path)
    parser.add_argument("--output-dir", default="experiments/hidden_memory_intervention", type=pathlib.Path)
    parser.add_argument("--target-sample-index", type=int, default=0)
    parser.add_argument("--target-local-chunk-index", type=int, default=2)
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

    _intervention._maybe_install_checkpoint_compat(args.checkpoint_dir)
    model = _intervention._load_model(config, args.checkpoint_dir)

    target_sample = _intervention._to_jax(dataset[args.target_sample_index])
    target_chunk_id = int(np.asarray(target_sample["chunks"]["chunk_ids"][args.target_local_chunk_index]).item())
    if target_chunk_id < 0 or not bool(np.asarray(target_sample["chunks"]["mask"][args.target_local_chunk_index]).item()):
        raise ValueError("Target local chunk is padded/invalid.")

    target_episode_id = _intervention._sample_episode_id(dataset, args.target_sample_index)
    hidden_by_chunk, observation_by_chunk, _ = _intervention._compute_episode_rollout(
        model,
        dataset,
        target_episode_id,
        stop_after_chunk_id=target_chunk_id,
    )
    observation = observation_by_chunk[target_chunk_id]
    hidden = hidden_by_chunk[target_chunk_id][None, :]
    memory_cond = model._temporal_condition_from_hidden(hidden, jnp.ones((1,), dtype=jnp.bool_))

    actions = []
    labels = []
    for i in range(args.num_noise_samples):
        noise_seed = args.seed + i
        noise = jax.random.normal(
            jax.random.key(noise_seed),
            (1, model.action_horizon, model.action_dim),
            dtype=observation.state.dtype,
        )
        action = model._sample_actions_with_memory(
            jax.random.key(args.seed + 10_000 + i),
            observation,
            num_steps=args.num_steps,
            noise=noise,
            memory_cond=memory_cond,
        )
        actions.append(np.asarray(action[0], dtype=np.float32))
        labels.append(f"noise_seed_{noise_seed}")

    ref = actions[0]
    rows = []
    for label, action in zip(labels, actions, strict=True):
        row = {
            "variant": label,
            "reference": labels[0],
            "target_episode_id": target_episode_id,
            "target_sample_index": args.target_sample_index,
            "target_local_chunk_index": args.target_local_chunk_index,
            "target_chunk_id": target_chunk_id,
        }
        row.update(_metrics(action, ref))
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "noise_baseline_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.output_dir / "noise_baseline_results.npz"
    np.savez_compressed(
        npz_path,
        actions=np.stack(actions, axis=0),
        variant_names=np.asarray(labels),
        target_episode_id=np.asarray(target_episode_id, dtype=np.int32),
        target_sample_index=np.asarray(args.target_sample_index, dtype=np.int32),
        target_local_chunk_index=np.asarray(args.target_local_chunk_index, dtype=np.int32),
        target_chunk_id=np.asarray(target_chunk_id, dtype=np.int32),
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    for row in rows:
        print(
            f"{row['variant']:>16s} mean_l2={row['mean_l2_per_step']:.6f} "
            f"first_l2={row['first_action_l2']:.6f} max_abs={row['max_abs_delta']:.6f} "
            f"cosine={row['cosine']:.6f}"
        )


if __name__ == "__main__":
    main()
