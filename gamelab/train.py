"""The AlphaZero loop: self-play -> replay buffer -> gradient steps -> evaluate.

Each iteration writes one JSON line to `runs/<name>/metrics.jsonl` (see
`gamelab.plotting`) and checkpoints go to `checkpoints/<name>/`.
"""

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gamelab.agents import Agent, MCTSAgent, RandomAgent
from gamelab.evaluate import evaluate_against
from gamelab.game import Game
from gamelab.network import NetworkEvaluator, get_device, load_checkpoint, save_checkpoint
from gamelab.replay import ReplayBuffer
from gamelab.selfplay import SelfPlayConfig, SelfPlayPool


@dataclass
class TrainConfig:
    name: str = "default"
    iterations: int = 100
    games_per_iteration: int = 128
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    num_workers: int = 0
    """Self-play processes. 0 plays in the training process, which is usually
    best with one GPU and a fast game; add workers when the game logic is the
    bottleneck."""
    buffer_capacity: int = 200_000
    min_buffer_size: int = 2_000
    """No training until the buffer holds this many samples."""
    batch_size: int = 256
    train_steps_per_iteration: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    eval_every: int = 5
    """Evaluate against the baselines every N iterations (0 = never)."""
    eval_games: int = 40
    eval_simulations: int | None = None
    """MCTS simulations for the evaluated agent; defaults to the self-play setting."""
    checkpoint_every: int = 10
    device: str | None = None
    """Training device; None = CUDA if available."""
    selfplay_device: str | None = None
    """Self-play inference device; None = same as `device`."""
    seed: int = 0
    runs_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"


class Trainer:
    def __init__(
        self,
        game_factory: Callable[[], Game],
        net: nn.Module,
        config: TrainConfig | None = None,
        baselines: dict[str, Agent] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        """`game_factory` must be picklable (e.g. the Game class) when using workers.
        `baselines` default to a random agent."""
        self.game_factory = game_factory
        self.config = config = config or TrainConfig()
        self.device = get_device(config.device)
        self.net = net.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.baselines = baselines if baselines is not None else {"random": RandomAgent(seed=config.seed)}
        g = game_factory()
        self.buffer = ReplayBuffer(config.buffer_capacity, g.observation_shape, g.action_size, g.num_players)
        self.rng = np.random.default_rng(config.seed)
        self.iteration = 0
        self.run_dir = Path(config.runs_dir) / config.name
        self.checkpoint_dir = Path(config.checkpoints_dir) / config.name

    # --- Pieces --------------------------------------------------------------

    def train_step(self, observations, policies, values) -> dict[str, float]:
        self.net.train()
        obs = torch.from_numpy(observations).to(self.device)
        target_pi = torch.from_numpy(policies).to(self.device)
        target_v = torch.from_numpy(values).to(self.device)
        logits, v = self.net(obs)
        policy_loss = -(target_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(v, target_v)
        loss = policy_loss + self.config.value_loss_weight * value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "loss": loss.item()}

    def train_epoch(self) -> dict[str, float]:
        """`train_steps_per_iteration` steps on random batches; mean losses."""
        losses = [
            self.train_step(*self.buffer.sample(self.config.batch_size, self.rng))
            for _ in range(self.config.train_steps_per_iteration)
        ]
        return {k: float(np.mean([l[k] for l in losses])) for k in losses[0]}

    def agent(self) -> MCTSAgent:
        """The current network as a greedy MCTS player."""
        sims = self.config.eval_simulations or self.config.selfplay.mcts.num_simulations
        mcts = dataclasses.replace(self.config.selfplay.mcts, num_simulations=sims, dirichlet_epsilon=0.0)
        return MCTSAgent(NetworkEvaluator(self.net, self.device), mcts, name="net", seed=self.config.seed)

    def evaluate(self) -> dict[str, float]:
        results = evaluate_against(
            self.game_factory, self.agent(), self.baselines, self.config.eval_games,
            self.rng, self.config.selfplay.max_moves,
        )
        metrics = {}
        for name, r in results.items():
            metrics[f"eval/{name}/score"] = r.score
            metrics[f"eval/{name}/win_rate"] = r.wins / r.games
        return metrics

    def save(self, path: Path) -> None:
        save_checkpoint(
            path, self.net, self.optimizer,
            iteration=self.iteration, config=dataclasses.asdict(self.config),
        )

    def load(self, path: str | Path) -> None:
        """Resume from a checkpoint saved by `save` (network, optimizer, iteration)."""
        _, data = load_checkpoint(path, self.net, self.optimizer, map_location=self.device)
        self.iteration = int(data.get("iteration", 0))

    # --- Loop ----------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        metrics_path = self.run_dir / "metrics.jsonl"
        with SelfPlayPool(
            self.game_factory, self.net, cfg.selfplay, cfg.num_workers,
            cfg.selfplay_device or str(self.device),
        ) as pool:
            while self.iteration < cfg.iterations:
                metrics = self.run_iteration(pool)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(_summary(metrics), flush=True)
                if cfg.checkpoint_every and self.iteration % cfg.checkpoint_every == 0:
                    self.save(self.checkpoint_dir / f"iter_{self.iteration:05d}.pt")
                self.save(self.checkpoint_dir / "latest.pt")

    def run_iteration(self, pool: SelfPlayPool) -> dict:
        cfg = self.config
        t0 = time.perf_counter()
        records = pool.play(cfg.games_per_iteration, self.net, seed=cfg.seed * 100_003 + self.iteration)
        for r in records:
            self.buffer.add_game(r)
        t1 = time.perf_counter()
        losses = self.train_epoch() if len(self.buffer) >= cfg.min_buffer_size else {}
        t2 = time.perf_counter()
        self.iteration += 1

        lengths = [r.length for r in records]
        decided = [r.winner for r in records if r.winner is not None]
        metrics = {
            "iteration": self.iteration,
            "time": time.time(),
            "selfplay/games": len(records),
            "selfplay/mean_length": float(np.mean(lengths)),
            "selfplay/draw_rate": 1 - len(decided) / len(records),
            "selfplay/seconds": t1 - t0,
            "train/seconds": t2 - t1,
            "buffer/size": len(self.buffer),
            "buffer/total_added": self.buffer.total_added,
            **{f"train/{k}": v for k, v in losses.items()},
        }
        if cfg.eval_every and self.iteration % cfg.eval_every == 0:
            t3 = time.perf_counter()
            metrics.update(self.evaluate())
            metrics["eval/seconds"] = time.perf_counter() - t3
        return metrics


def _summary(m: dict) -> str:
    parts = [f"iter {m['iteration']:4d}", f"len {m['selfplay/mean_length']:5.1f}", f"buf {m['buffer/size']}"]
    if "train/loss" in m:
        parts.append(f"loss {m['train/loss']:.3f} (pi {m['train/policy_loss']:.3f} v {m['train/value_loss']:.3f})")
    parts += [f"{k.split('/')[1]} {v:.2f}" for k, v in m.items() if k.endswith("/score")]
    parts.append(f"{m['selfplay/seconds'] + m['train/seconds']:.1f}s")
    return " | ".join(parts)
