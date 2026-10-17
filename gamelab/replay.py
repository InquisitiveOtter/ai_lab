"""A fixed-capacity replay buffer of (observation, policy target, value target)."""

from pathlib import Path

import numpy as np


class ReplayBuffer:
    """Ring buffer: once full, the oldest samples are overwritten."""

    def __init__(self, capacity: int, observation_shape, action_size: int, num_players: int):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.policies = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros((capacity, num_players), dtype=np.float32)
        self.size = 0
        self.pos = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add(self, observations, policies, values) -> None:
        """Add a batch of samples (leading dimension = number of samples)."""
        n = len(observations)
        self.total_added += n
        if n > self.capacity:  # only the newest `capacity` samples would survive anyway
            observations, policies, values = observations[-self.capacity:], policies[-self.capacity:], values[-self.capacity:]
            n = self.capacity
        idx = (self.pos + np.arange(n)) % self.capacity
        self.observations[idx] = observations
        self.policies[idx] = policies
        self.values[idx] = values
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def add_game(self, record) -> None:
        """Add every sample from a `gamelab.selfplay.GameRecord`."""
        self.add(record.observations, record.policies, record.values)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        """A uniformly random batch `(observations, policies, values)`."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return self.observations[idx], self.policies[idx], self.values[idx]

    def save(self, path: str | Path) -> None:
        """Save samples oldest-first, so `load` can restore them with `add`."""
        start = self.pos if self.size == self.capacity else 0
        order = (start + np.arange(self.size)) % self.capacity
        np.savez_compressed(
            path, observations=self.observations[order], policies=self.policies[order],
            values=self.values[order], total_added=self.total_added,
        )

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None) -> "ReplayBuffer":
        """Load a saved buffer; with a smaller `capacity`, only the newest samples are kept."""
        data = np.load(path)
        obs, pol, val = data["observations"], data["policies"], data["values"]
        buf = cls(capacity or max(len(obs), 1), obs.shape[1:], pol.shape[1], val.shape[1])
        if len(obs):
            buf.add(obs, pol, val)
        buf.total_added = int(data["total_added"])
        return buf
