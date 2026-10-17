"""Policy/value networks and the bridge between a network and MCTS.

Network contract (any `nn.Module` can be used if it follows it):

    forward(obs: (B, *observation_shape)) -> (policy_logits: (B, action_size),
                                              values: (B, num_players) in [-1, 1])

Values are ordered *relative to the player to move*: `values[:, k]` is the value
for player `(current_player + k) % num_players`. This matches observations
encoded from the mover's perspective. `NetworkEvaluator` converts to absolute
player order for MCTS, and self-play stores targets in the relative order.
"""

import io
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gamelab.game import Game


def get_device(device: str | torch.device | None = None) -> torch.device:
    """The requested device, or CUDA if available (warning when it falls back to CPU)."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    warnings.warn("CUDA is not available; using CPU", stacklevel=2)
    return torch.device("cpu")


def to_relative(values: np.ndarray, player: int) -> np.ndarray:
    """Absolute-player values -> values relative to `player`."""
    return np.roll(values, -player, axis=-1)


def to_absolute(values: np.ndarray, player: int) -> np.ndarray:
    """Values relative to `player` -> absolute-player values."""
    return np.roll(values, player, axis=-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(inplace=True),
            nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class PolicyValueNet(nn.Module):
    """A general-purpose policy/value network.

    3-D observations (C, H, W) get a convolutional ResNet trunk; anything else is
    flattened and goes through a residual MLP. Games with structure that neither
    fits (graphs, sets of cards) should define their own module with the same
    forward contract.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        num_players: int = 2,
        width: int = 128,
        blocks: int = 4,
    ):
        super().__init__()
        self.config = dict(
            observation_shape=tuple(observation_shape), action_size=action_size,
            num_players=num_players, width=width, blocks=blocks,
        )
        if len(observation_shape) == 3:
            c, h, w = observation_shape
            self.trunk = nn.Sequential(
                nn.Conv2d(c, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                *[ResidualConvBlock(width) for _ in range(blocks)],
            )
            self.policy_head = nn.Sequential(
                nn.Conv2d(width, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True),
                nn.Flatten(), nn.Linear(2 * h * w, action_size),
            )
            self.value_head = nn.Sequential(
                nn.Conv2d(width, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True),
                nn.Flatten(), nn.Linear(h * w, width), nn.ReLU(inplace=True),
                nn.Linear(width, num_players), nn.Tanh(),
            )
        else:
            features = int(np.prod(observation_shape))
            self.trunk = nn.Sequential(
                nn.Flatten(), nn.Linear(features, width), nn.ReLU(inplace=True),
                *[ResidualMLPBlock(width) for _ in range(blocks)],
            )
            self.policy_head = nn.Linear(width, action_size)
            self.value_head = nn.Sequential(
                nn.Linear(width, width), nn.ReLU(inplace=True),
                nn.Linear(width, num_players), nn.Tanh(),
            )

    @classmethod
    def for_game(cls, game: Game | type[Game], **kwargs) -> "PolicyValueNet":
        return cls(game.observation_shape, game.action_size, game.num_players, **kwargs)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)


class NetworkEvaluator:
    """Runs a policy/value network for MCTS: batched, masked to legal actions."""

    def __init__(self, net: nn.Module, device: str | torch.device | None = None):
        self.device = get_device(device)
        self.net = net.to(self.device)

    @torch.inference_mode()
    def evaluate(self, states: list[Game]) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        obs = torch.from_numpy(np.stack([s.encode() for s in states])).float().to(self.device)
        mask = torch.from_numpy(np.stack([s.legal_mask() for s in states])).to(self.device)
        logits, values = self.net(obs)
        priors = torch.softmax(logits.float().masked_fill(~mask, float("-inf")), dim=1)
        priors, values = priors.cpu().numpy(), values.float().cpu().numpy()
        values = np.stack([to_absolute(v, s.current_player()) for v, s in zip(values, states)])
        return priors, values


def save_checkpoint(path: str | Path, net: nn.Module, optimizer=None, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": net.state_dict(), "net_config": getattr(net, "config", None), **extra}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, net: nn.Module | None = None, optimizer=None, map_location="cpu"):
    """Load a checkpoint into `net` (built from the stored config if omitted).

    Returns `(net, checkpoint_dict)`.
    """
    data = torch.load(path, map_location=map_location, weights_only=True)
    if net is None:
        if data.get("net_config") is None:
            raise ValueError(f"{path} has no net_config; pass the network to load into")
        net = PolicyValueNet(**data["net_config"])
    net.load_state_dict(data["model"])
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    return net, data


def serialize(obj) -> bytes:
    """torch.save to bytes, for shipping modules and weights to worker processes."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def deserialize(data: bytes, weights_only: bool = True):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=weights_only)
