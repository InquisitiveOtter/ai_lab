"""The interface every game implements.

A `Game` instance *is* a game state. The core never looks inside it; it only
uses the methods below. Actions are integers in a fixed action space
`[0, action_size)`, so a network can have one policy output per action.

Beyond simple two-player board games, the interface covers:

- any number of players (`num_players`),
- a player moving several times in a row (`current_player()` need not alternate),
- randomness (dice, card draws) as explicit *chance nodes*: when
  `current_player()` returns `CHANCE`, the next action is drawn from
  `chance_outcomes()` instead of being chosen by a player.
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

CHANCE = -1
"""Returned by `Game.current_player()` when the next action is a random event."""


class Game(ABC):
    num_players: int = 2
    action_size: int
    """Number of distinct actions across all states (the policy head's width)."""
    observation_shape: tuple[int, ...]
    """Shape of `encode()`'s output, e.g. (channels, height, width) or (features,)."""

    # --- Required -----------------------------------------------------------

    @abstractmethod
    def current_player(self) -> int:
        """Index of the player to move, in `[0, num_players)`, or `CHANCE`."""

    @abstractmethod
    def legal_actions(self) -> list[int]:
        """Actions available in this state. Undefined once the game is over."""

    @abstractmethod
    def apply(self, action: int) -> None:
        """Play `action`, mutating this state in place."""

    @abstractmethod
    def clone(self) -> Self:
        """An independent copy of this state."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """True once the game is over."""

    @abstractmethod
    def winner(self) -> int | None:
        """The winning player once the game is over, or None (draw, or not over)."""

    @abstractmethod
    def encode(self) -> np.ndarray:
        """The state as a float32 array of `observation_shape`, for the network.

        Encode from the perspective of `current_player()` (e.g. "my pieces" and
        "opponent pieces" planes): the network's value output is ordered
        relative to the player to move, see `gamelab.network`.
        """

    # --- Optional -----------------------------------------------------------

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """(action, probability) pairs for a chance node. Only called when
        `current_player() == CHANCE`."""
        raise NotImplementedError(f"{type(self).__name__} has no chance nodes")

    def returns(self) -> np.ndarray:
        """Final outcome for each player (indexed by player), in [-1, 1].

        Default: +1 for the winner, -1 for everyone else, 0 for all on a draw.
        Override for score-based outcomes.
        """
        out = np.zeros(self.num_players, dtype=np.float32)
        w = self.winner()
        if w is not None:
            out[:] = -1.0
            out[w] = 1.0
        return out

    def symmetries(
        self, observation: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Equivalent (observation, policy) pairs used to augment training data.

        Default: no symmetries. Board games can return rotations/reflections.
        """
        return [(observation, policy)]

    # --- Provided -----------------------------------------------------------

    def step(self, action: int) -> Self:
        """A new state with `action` played; this state is left unchanged."""
        nxt = self.clone()
        nxt.apply(action)
        return nxt

    def is_chance(self) -> bool:
        return self.current_player() == CHANCE

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        mask[self.legal_actions()] = True
        return mask


def sample_chance(state: Game, rng: np.random.Generator) -> int:
    """Draw an outcome at a chance node."""
    actions, probs = zip(*state.chance_outcomes())
    return actions[rng.choice(len(actions), p=np.asarray(probs) / np.sum(probs))]


def resolve_chance(state: Game, rng: np.random.Generator) -> Game:
    """Play random outcomes until a player is to move or the game ends."""
    while not state.is_terminal() and state.is_chance():
        state = state.step(sample_chance(state, rng))
    return state
