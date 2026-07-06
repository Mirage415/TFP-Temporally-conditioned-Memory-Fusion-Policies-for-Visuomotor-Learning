import argparse
import csv
import dataclasses
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.temporal_memory_loader as _temporal_memory_loader


def _to_jax(tree, *, add_batch: bool = False):
    if add_batch:
        return jax.tree.map(lambda x: jnp.asarray(np.asarray(x))[None, ...], tree)
    return jax.tree.map(lambda x: jnp.asarray(np.asarray(x)), tree)


def _episode_windows(dataset, episode_id: int) -> list[tuple[int, int]]:
    windows = getattr(dataset, "_windows", None)
    if windows is None:
        raise ValueError("Temporal cache does not expose window metadata.")
    result = [
        (sample_index, int(window.start_chunk_id))
        for sample_index, window in enumerate(windows)
        if int(window.episode_index) == int(episode_id)
    ]
    if not result:
        raise ValueError(f"No windows found for episode_id={episode_id}.")
    return sorted(result, key=lambda item: item[1])


def _load_model(config: _config.TrainConfig, checkpoint_dir: pathlib.Path):
    params_dir = checkpoint_dir / "params" if (checkpoint_dir / "params").exists() else checkpoint_dir
    if not params_dir.exists():
        raise FileNotFoundError(f"Could not find params directory at {params_dir}")
    return config.model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))


def _checkpoint_param_names(checkpoint_dir: pathlib.Path) -> list[str]:
    params_dir = checkpoint_dir / "params" if (checkpoint_dir / "params").exists() else checkpoint_dir
    metadata_path = params_dir / "array_metadatas" / "process_0"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text())
    return [item["array_metadata"]["param_name"] for item in metadata.get("array_metadatas", [])]


def _install_head_module(module_name: str, repo_path: str) -> None:
    source = subprocess.check_output(
        ["git", "show", f"HEAD:{repo_path}"],
        text=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
    )
    tmp_dir = tempfile.mkdtemp(prefix="openpi_legacy_adaln_")
    module_path = pathlib.Path(tmp_dir) / pathlib.Path(repo_path).name
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if "." in module_name:
        parent_name, attr_name = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr_name, module)


def _maybe_install_checkpoint_compat(checkpoint_dir: pathlib.Path) -> None:
    names = _checkpoint_param_names(checkpoint_dir)
    has_legacy_adaln = any(".memory_adaln_proj." in name for name in names)
    has_new_router = any(".memory_router." in name for name in names)
    if has_legacy_adaln and not has_new_router:
        print("[compat] Detected legacy AdaLN temporal-memory checkpoint; using HEAD gemma/pi0_adaln implementations.")
        _install_head_module("openpi.models.gemma", "src/openpi/models/gemma.py")
        _install_head_module("openpi.models.pi0_adaln", "src/openpi/models/pi0_adaln.py")


def _sample_episode_id(dataset, sample_index: int) -> int:
    windows = getattr(dataset, "_windows", None)
    if windows is not None:
        return int(windows[sample_index].episode_index)
    sample = dataset[sample_index]
    return int(np.asarray(sample["episode_id"]).item())


def _compute_episode_rollout(model, dataset, episode_id: int, *, stop_after_chunk_id: int | None = None):
    hidden = None
    hidden_by_chunk = {}
    observation_by_chunk = {}
    sample_by_chunk = {}

    for sample_index, _ in _episode_windows(dataset, episode_id):
        sample = _to_jax(dataset[sample_index])
        if hidden is None:
            hidden = model.ltc_encoder.initial_state(1, dtype=sample["chunks"]["state"].dtype)

        num_chunks = sample["chunks"]["state"].shape[0]
        for local_chunk_index in range(num_chunks):
            chunk_id = int(np.asarray(sample["chunks"]["chunk_ids"][local_chunk_index]).item())
            if stop_after_chunk_id is not None and chunk_id > stop_after_chunk_id:
                return hidden_by_chunk, observation_by_chunk, sample_by_chunk
            mask = sample["chunks"]["mask"][local_chunk_index : local_chunk_index + 1]
            if chunk_id < 0 or not bool(np.asarray(mask[0]).item()):
                continue
            obs_t = model._prepare_temporal_observation(
                jax.tree.map(lambda x: x[None, local_chunk_index], sample["chunks"]["images"]),
                jax.tree.map(lambda x: x[None, local_chunk_index], sample["chunks"]["image_masks"]),
                sample["chunks"]["state"][None, local_chunk_index],
                _to_jax(sample["prompt"], add_batch=True),
                None,
                train=False,
            )
            update_result = model._update_temporal_hidden(
                obs_t,
                hidden,
                sample["chunks"]["delta_t"][local_chunk_index : local_chunk_index + 1],
                mask=mask,
            )
            hidden = update_result[0]
            hidden_by_chunk[chunk_id] = jax.lax.stop_gradient(hidden[0])
            observation_by_chunk[chunk_id] = obs_t
            sample_by_chunk[chunk_id] = (sample_index, local_chunk_index)

    if hidden is None or not hidden_by_chunk:
        raise ValueError(f"No valid chunks found while replaying episode_id={episode_id}.")
    return hidden_by_chunk, observation_by_chunk, sample_by_chunk


def _select_chunk(chunk_ids: list[int], desired: int) -> int:
    if desired in chunk_ids:
        return desired
    before = [chunk_id for chunk_id in chunk_ids if chunk_id <= desired]
    return before[-1] if before else chunk_ids[0]


def _build_variants(target_hidden_by_chunk, other_hidden_by_chunk, target_chunk_id: int, seed: int):
    target_chunks = sorted(target_hidden_by_chunk)
    variants = {
        "zero": jnp.zeros_like(target_hidden_by_chunk[target_chunk_id]),
        "target_normal": target_hidden_by_chunk[target_chunk_id],
        "target_episode_first": target_hidden_by_chunk[target_chunks[0]],
    }
    if len(target_chunks) > 2:
        variants["target_episode_mid_so_far"] = target_hidden_by_chunk[target_chunks[len(target_chunks) // 2]]

    if other_hidden_by_chunk:
        other_chunks = sorted(other_hidden_by_chunk)
        variants["other_episode_same_or_nearest"] = other_hidden_by_chunk[_select_chunk(other_chunks, target_chunk_id)]
        variants["other_episode_first"] = other_hidden_by_chunk[other_chunks[0]]

    all_hidden = list(target_hidden_by_chunk.values()) + list(other_hidden_by_chunk.values())
    hidden_np = np.asarray(jnp.stack(all_hidden, axis=0), dtype=np.float32)
    variants["valid_hidden_mean"] = jnp.asarray(hidden_np.mean(axis=0), dtype=all_hidden[0].dtype)
    rng = np.random.default_rng(seed)
    variants["random_matched_stats"] = jnp.asarray(
        rng.normal(hidden_np.mean(axis=0), hidden_np.std(axis=0) + 1e-6).astype(np.float32),
        dtype=all_hidden[0].dtype,
    )
    return variants


def _metrics(action, ref_action):
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


def main():
    parser = argparse.ArgumentParser(description="Run same-observation hidden-memory intervention on training data.")
    parser.add_argument("--config-name", default="tfp_a1")
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--temporal-cache-dir", default="assets/tfp_a1/temporal_cache", type=pathlib.Path)
    parser.add_argument("--output-dir", default="experiments/hidden_memory_intervention", type=pathlib.Path)
    parser.add_argument("--target-sample-index", type=int, default=0)
    parser.add_argument("--target-local-chunk-index", type=int, default=2)
    parser.add_argument("--other-episode-id", type=int, default=None)
    parser.add_argument("--full-episode-variants", action="store_true")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, temporal_cache_dir=str(args.temporal_cache_dir))
    dataset = _temporal_memory_loader.CachedTemporalMemoryDataset(
        args.temporal_cache_dir,
        config.model.tbptt_num_chunks,
        config.model.tbptt_chunk_len,
    )

    _maybe_install_checkpoint_compat(args.checkpoint_dir)
    model = _load_model(config, args.checkpoint_dir)
    target_sample = _to_jax(dataset[args.target_sample_index])
    if args.target_local_chunk_index >= target_sample["chunks"]["chunk_ids"].shape[0]:
        raise ValueError(
            "target local chunk index "
            f"{args.target_local_chunk_index} out of range for "
            f"{target_sample['chunks']['chunk_ids'].shape[0]} chunks"
        )
    target_chunk_id = int(np.asarray(target_sample["chunks"]["chunk_ids"][args.target_local_chunk_index]).item())
    if target_chunk_id < 0 or not bool(np.asarray(target_sample["chunks"]["mask"][args.target_local_chunk_index]).item()):
        raise ValueError(
            f"target sample {args.target_sample_index} local chunk {args.target_local_chunk_index} is padded/invalid."
        )

    target_episode_id = _sample_episode_id(dataset, args.target_sample_index)
    stop_after_chunk_id = None if args.full_episode_variants else target_chunk_id
    target_hidden_by_chunk, target_observation_by_chunk, target_sample_by_chunk = _compute_episode_rollout(
        model,
        dataset,
        target_episode_id,
        stop_after_chunk_id=stop_after_chunk_id,
    )
    observation = target_observation_by_chunk[target_chunk_id]

    other_episode_id = args.other_episode_id
    if other_episode_id is None:
        episode_ids = sorted({int(window.episode_index) for window in getattr(dataset, "_windows")})
        other_episode_id = next((episode_id for episode_id in episode_ids if episode_id != target_episode_id), None)
    other_hidden_by_chunk = {}
    if other_episode_id is not None:
        other_hidden_by_chunk, _, _ = _compute_episode_rollout(
            model,
            dataset,
            other_episode_id,
            stop_after_chunk_id=stop_after_chunk_id,
        )

    variants = _build_variants(
        target_hidden_by_chunk,
        other_hidden_by_chunk,
        target_chunk_id,
        args.seed + 100,
    )

    noise = jax.random.normal(
        jax.random.key(args.seed),
        (1, model.action_horizon, model.action_dim),
        dtype=observation.state.dtype,
    )

    actions = {}
    memory_conds = {}
    for name, hidden in variants.items():
        hidden = hidden[None, :]
        memory_cond = model._temporal_condition_from_hidden(hidden, jnp.ones((1,), dtype=jnp.bool_))
        action = model._sample_actions_with_memory(
            jax.random.key(args.seed + 1),
            observation,
            num_steps=args.num_steps,
            noise=noise,
            memory_cond=memory_cond,
        )
        actions[name] = np.asarray(action[0], dtype=np.float32)
        memory_conds[name] = np.asarray(memory_cond[0], dtype=np.float32)

    ref_name = "target_normal"
    rows = []
    for name, action in actions.items():
        row = {
            "variant": name,
            "reference": ref_name,
            "target_episode_id": target_episode_id,
            "target_sample_index": args.target_sample_index,
            "target_local_chunk_index": args.target_local_chunk_index,
            "target_chunk_id": target_chunk_id,
            "other_episode_id": -1 if other_episode_id is None else other_episode_id,
        }
        row.update(_metrics(action, actions[ref_name]))
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.output_dir / "results.npz"
    np.savez_compressed(
        npz_path,
        actions=np.stack([actions[name] for name in actions], axis=0),
        hiddens=np.stack([np.asarray(variants[name], dtype=np.float32) for name in actions], axis=0),
        memory_conds=np.stack([memory_conds[name] for name in actions], axis=0),
        variant_names=np.asarray(list(actions.keys())),
        target_state=np.asarray(observation.state[0], dtype=np.float32),
        target_episode_id=np.asarray(target_episode_id, dtype=np.int32),
        target_sample_index=np.asarray(args.target_sample_index, dtype=np.int32),
        target_local_chunk_index=np.asarray(args.target_local_chunk_index, dtype=np.int32),
        target_chunk_id=np.asarray(target_chunk_id, dtype=np.int32),
        target_source_sample=np.asarray(target_sample_by_chunk[target_chunk_id][0], dtype=np.int32),
        target_source_local_chunk=np.asarray(target_sample_by_chunk[target_chunk_id][1], dtype=np.int32),
        other_episode_id=np.asarray(-1 if other_episode_id is None else other_episode_id, dtype=np.int32),
        checkpoint_dir=np.asarray(str(args.checkpoint_dir)),
        config_name=np.asarray(args.config_name),
    )
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    for row in rows:
        print(
            f"{row['variant']:>24s} mean_l2={row['mean_l2_per_step']:.6f} "
            f"first_l2={row['first_action_l2']:.6f} max_abs={row['max_abs_delta']:.6f} "
            f"cosine={row['cosine']:.6f}"
        )


if __name__ == "__main__":
    main()
