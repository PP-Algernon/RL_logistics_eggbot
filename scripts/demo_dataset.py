"""Read successful Eggtart demonstrations with a reproducible episode split."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch


def load_dataset(filepath, device="cpu", train_split=0.9, seed=42, *, split_metadata=None,
                 expected_env=None):
    """Reuse a saved split only after verifying dataset identity and episode coverage."""
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1")
    with h5py.File(filepath, "r") as f:
        obs, actions = f["obs"][:], f["action"][:]
        if "episode_lengths" not in f:
            raise ValueError("episode_lengths is required for an episode-level validation split")
        lengths = f["episode_lengths"][:]
        if not f.attrs.get("successful_only", False):
            raise ValueError("Use a dataset marked successful_only=True")
        env_name = str(f.attrs.get("env_name", ""))
    if expected_env and env_name and env_name != expected_env:
        raise ValueError(f"Dataset task {env_name} does not match {expected_env}")
    if obs.ndim != 2 or actions.ndim != 2 or obs.shape[0] != actions.shape[0]:
        raise ValueError("obs/action must be 2D arrays with the same number of samples")
    if obs.shape[1:] != (44,) or actions.shape[1:] != (9,):
        raise ValueError(f"Expected Eggtart obs/action dimensions 44/9, got {obs.shape}/{actions.shape}")
    if not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError("Dataset contains NaN/Inf")
    if (lengths.ndim != 1 or len(lengths) < 2 or not np.issubdtype(lengths.dtype, np.integer)
            or (lengths <= 0).any() or lengths.sum() != len(obs)):
        raise ValueError("Need at least two complete episodes with valid episode_lengths")

    digest = hashlib.sha256(env_name.encode())
    for array in (obs, actions, lengths):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    fingerprint = digest.hexdigest()
    if split_metadata:
        saved_hash = split_metadata.get("dataset_sha256")
        if saved_hash and saved_hash != fingerprint:
            raise ValueError("Demonstration dataset differs from the checkpoint dataset")
        if not saved_hash and split_metadata.get("dataset_path") != str(Path(filepath).resolve()):
            raise ValueError("Checkpoint has no dataset fingerprint; use its original dataset path")
        train_episodes = np.asarray(split_metadata["train_episode_indices"])
        val_episodes = np.asarray(split_metadata["val_episode_indices"])
        all_episodes = np.concatenate((train_episodes, val_episodes))
        if (not len(train_episodes) or not len(val_episodes)
                or not np.issubdtype(all_episodes.dtype, np.integer)
                or not np.array_equal(np.sort(all_episodes), np.arange(len(lengths)))):
            raise ValueError("Saved split must cover every episode exactly once")
        seed = split_metadata["split_seed"]
    else:
        order = np.random.default_rng(seed).permutation(len(lengths))
        n_train = min(len(lengths) - 1, max(1, int(len(lengths) * train_split)))
        train_episodes, val_episodes = order[:n_train], order[n_train:]
    is_train = np.zeros(len(lengths), dtype=bool)
    is_train[train_episodes] = True
    mask = np.repeat(is_train, lengths)
    result = tuple(torch.as_tensor(a, dtype=torch.float32, device=device) for a in (
        obs[mask], actions[mask], obs[~mask], actions[~mask]
    ))
    metadata = {
        "env_name": env_name, "dataset_path": str(Path(filepath).resolve()),
        "dataset_sha256": fingerprint, "split_seed": seed,
        "train_episode_indices": train_episodes.tolist(), "val_episode_indices": val_episodes.tolist(),
        "train_samples": int(mask.sum()), "val_samples": int((~mask).sum()),
    }
    print(f"Dataset: {len(lengths):,} successful episodes / {len(obs):,} samples")
    print(f"Split by episode: train={len(train_episodes):,} ({mask.sum():,} samples), "
          f"validation={len(val_episodes):,} ({(~mask).sum():,} samples)")
    return (*result, metadata)
