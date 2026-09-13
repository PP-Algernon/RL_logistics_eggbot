"""Episode accounting for vectorized grasp evaluation; no simulator imports."""
from __future__ import annotations

from collections import Counter

import torch


class EpisodeStatistics:
    """Assign fixed episode quotas so early completions cannot bias the sample."""

    def __init__(self, num_envs: int, num_episodes: int, device: str):
        if not 0 < num_envs <= num_episodes:
            raise ValueError("Require 0 < num_envs <= num_episodes")
        self.requested = num_episodes
        self.quotas = torch.full((num_envs,), num_episodes // num_envs, device=device, dtype=torch.long)
        self.quotas[:num_episodes % num_envs] += 1
        self.completed = torch.zeros_like(self.quotas)
        self.lengths = torch.zeros_like(self.quotas)
        self.returns = torch.zeros(num_envs, device=device)
        self.episodes: list[dict] = []

    @property
    def finished(self) -> bool:
        return len(self.episodes) == self.requested

    def update(self, rewards: torch.Tensor, dones: torch.Tensor,
               term_flags: dict[str, torch.Tensor], failure_terms: set[str], step_dt: float):
        active = self.completed < self.quotas
        self.lengths += active.long()
        self.returns += torch.where(active, rewards, 0.0)
        ids = (active & dones.bool()).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        flags = {name: values[ids].bool().tolist() for name, values in term_flags.items()}
        lengths, returns = self.lengths[ids].tolist(), self.returns[ids].tolist()
        counts = self.completed[ids].tolist()
        for index, env_id in enumerate(ids.tolist()):
            reasons = [name for name, values in flags.items() if values[index]]
            failures = failure_terms.intersection(reasons)
            # Failure wins over simultaneous success; success on the last allowed
            # step is valid even if the timeout term fires on that same step.
            if failures:
                outcome = next((name for name in ("base_tipped", "target_dropped") if name in failures),
                               sorted(failures)[0])
            elif "lift_success" in reasons:
                outcome = "success"
            elif "time_out" in reasons:
                outcome = "time_out"
            else:
                outcome = "other_termination"
            self.episodes.append({
                "env_id": env_id, "episode_index": counts[index], "outcome": outcome,
                "termination_terms": reasons, "steps": lengths[index],
                "duration_s": lengths[index] * step_dt, "reward": returns[index],
            })
        self.completed[ids] += 1
        self.lengths[ids] = 0
        self.returns[ids] = 0.0

    def summary(self) -> dict:
        count = len(self.episodes)
        outcomes = Counter(episode["outcome"] for episode in self.episodes)
        successes = outcomes["success"]
        return {
            "requested_episodes": self.requested, "completed_episodes": count,
            "complete": self.finished, "successes": successes,
            "success_rate": successes / count if count else None,
            "outcomes": {name: outcomes[name] for name in
                         ("success", "time_out", "target_dropped", "base_tipped", "other_termination")}
                        | dict(outcomes),
            "mean_episode_duration_s": sum(e["duration_s"] for e in self.episodes) / count if count else None,
            "mean_episode_reward": sum(e["reward"] for e in self.episodes) / count if count else None,
        }
