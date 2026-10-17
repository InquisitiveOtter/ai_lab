"""Self-play: generating training data by having the network play itself.

Two levels of parallelism:

- Within a process, `play_games` runs `concurrent_games` games in lockstep so
  every MCTS simulation evaluates one leaf per game in a single batch.
- `SelfPlayPool` spreads games over worker processes. Workers are started with
  `spawn` (the only option on Windows), so the game factory and the network's
  class must be importable from a module, not defined in `__main__` or a notebook.
"""

import copy
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from gamelab.game import Game, resolve_chance
from gamelab.mcts import MCTS, Evaluator, MCTSConfig, select_action, visit_policy
from gamelab.network import NetworkEvaluator, deserialize, get_device, serialize, to_relative


@dataclass
class SelfPlayConfig:
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    temperature_moves: int = 10
    """Sample moves in proportion to visit counts for this many moves per game,
    then play the most-visited move."""
    concurrent_games: int = 32
    """Games per process played in lockstep; also the inference batch size."""
    max_moves: int | None = None
    """Cut games off after this many moves and score them as draws."""


@dataclass
class GameRecord:
    """Training samples from one game (including symmetry-augmented copies)."""

    observations: np.ndarray  # (N, *observation_shape)
    policies: np.ndarray  # (N, action_size), MCTS visit distributions
    values: np.ndarray  # (N, num_players), final returns relative to the player to move
    returns: np.ndarray  # (num_players,), final returns by absolute player
    length: int  # moves played by players (chance outcomes excluded)
    winner: int | None
    truncated: bool = False


class _Episode:
    def __init__(self, state: Game):
        self.state = state
        self.moves = 0
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []

    def finish(self, truncated: bool) -> GameRecord:
        s = self.state
        returns = np.zeros(s.num_players, dtype=np.float32) if truncated else s.returns().astype(np.float32)
        obs, pol, val = [], [], []
        for o, p, player in self.samples:
            obs.append(o)
            pol.append(p)
            val.append(to_relative(returns, player))
        a = s.action_size
        return GameRecord(
            observations=np.array(obs, dtype=np.float32).reshape(-1, *s.observation_shape),
            policies=np.array(pol, dtype=np.float32).reshape(-1, a),
            values=np.array(val, dtype=np.float32).reshape(-1, s.num_players),
            returns=returns,
            length=self.moves,
            winner=None if truncated else s.winner(),
            truncated=truncated,
        )


def play_games(
    game_factory: Callable[[], Game],
    evaluator: Evaluator,
    num_games: int,
    config: SelfPlayConfig | None = None,
    rng: np.random.Generator | None = None,
) -> list[GameRecord]:
    """Play `num_games` self-play games with MCTS guided by `evaluator`."""
    config = config or SelfPlayConfig()
    rng = rng if rng is not None else np.random.default_rng()
    mcts = MCTS(evaluator, config.mcts, rng)
    records: list[GameRecord] = []
    active: list[_Episode] = []
    started = 0
    while started < num_games or active:
        while len(active) < config.concurrent_games and started < num_games:
            active.append(_Episode(game_factory()))
            started += 1
        still_active = []
        for ep in active:
            ep.state = resolve_chance(ep.state, rng)
            truncated = config.max_moves is not None and ep.moves >= config.max_moves
            if ep.state.is_terminal() or truncated:
                records.append(ep.finish(truncated=not ep.state.is_terminal()))
            else:
                still_active.append(ep)
        active = still_active
        if not active:
            continue

        roots = mcts.search([ep.state for ep in active], add_noise=True)
        for ep, root in zip(active, roots):
            target = visit_policy(root, 1.0)
            for obs, pol in ep.state.symmetries(ep.state.encode(), target):
                ep.samples.append((obs, pol, ep.state.current_player()))
            temperature = 1.0 if ep.moves < config.temperature_moves else 0.0
            ep.state = ep.state.step(select_action(root, temperature, rng))
            ep.moves += 1
    return records


# --- Multi-process self-play -------------------------------------------------


class _Worker:
    """Holds a network copy and plays games with it. Lives in each worker process
    (or in the main process when the pool has no workers)."""

    def __init__(self, game_factory, model_bytes: bytes, device: str, config: SelfPlayConfig):
        self.game_factory = game_factory
        self.config = config
        self.version = None
        net = deserialize(model_bytes, weights_only=False)
        self.evaluator = NetworkEvaluator(net, device)

    def play(self, num_games: int, weights: bytes, version: int, seed: int) -> list[GameRecord]:
        if version != self.version:
            self.evaluator.net.load_state_dict(deserialize(weights))
            self.version = version
        return play_games(
            self.game_factory, self.evaluator, num_games, self.config, np.random.default_rng(seed)
        )


_worker: _Worker | None = None


def _init_worker(*args):
    global _worker
    import torch

    if args[2] == "cpu":
        torch.set_num_threads(1)  # many CPU workers: avoid oversubscribing cores
    _worker = _Worker(*args)


def _play_in_worker(*args):
    return _worker.play(*args)


class SelfPlayPool:
    """Self-play across `num_workers` processes (0 = in the calling process).

    Use as a context manager, or call `close()`.
    """

    def __init__(
        self,
        game_factory: Callable[[], Game],
        net,
        config: SelfPlayConfig | None = None,
        num_workers: int = 0,
        device: str | None = None,
    ):
        self.config = config or SelfPlayConfig()
        self.num_workers = num_workers
        self._version = 0
        device = str(get_device(device))
        model_bytes = serialize(copy.deepcopy(net).cpu())
        args = (game_factory, model_bytes, device, self.config)
        if num_workers == 0:
            self._local = _Worker(*args)
            self._executor = None
        else:
            self._local = None
            self._executor = ProcessPoolExecutor(
                num_workers, mp_context=mp.get_context("spawn"),
                initializer=_init_worker, initargs=args,
            )

    def play(self, num_games: int, net, seed: int = 0) -> list[GameRecord]:
        """Play `num_games` games with the current weights of `net`."""
        self._version += 1
        weights = serialize({k: v.detach().cpu() for k, v in net.state_dict().items()})
        if self._executor is None:
            return self._local.play(num_games, weights, self._version, seed)
        shares = [num_games // self.num_workers + (i < num_games % self.num_workers) for i in range(self.num_workers)]
        futures = [
            self._executor.submit(_play_in_worker, n, weights, self._version, seed * 1_000_003 + i)
            for i, n in enumerate(shares) if n > 0
        ]
        return [record for f in futures for record in f.result()]

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
