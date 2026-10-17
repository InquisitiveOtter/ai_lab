"""A tiny 3-player dice game that exercises chance nodes, more than two players,
and a player making several decisions in one turn.

Each turn: a die (1-3) is rolled (a chance node), then the player either moves
forward by the roll ("go", action 0) or moves 1 ("step", action 1). Rolling a 3
grants a second decision with the same roll before the turn passes. First to
reach `TARGET` wins. "go" is never worse, so the best policy is easy to check.
"""

import numpy as np

from gamelab import CHANCE, Game

TARGET = 8
GO, STEP = 0, 1
ROLL_ACTIONS = {1: 2, 2: 3, 3: 4}  # die face -> chance action id


class DiceRace(Game):
    num_players = 3
    action_size = 5  # 0 go, 1 step, 2-4 die faces 1-3
    observation_shape = (3 + 3 + 1,)

    def __init__(self):
        self.pos = [0, 0, 0]
        self.player = 0
        self.roll: int | None = None
        self.decisions_left = 0
        self._winner: int | None = None

    def current_player(self) -> int:
        return CHANCE if self.roll is None else self.player

    def chance_outcomes(self):
        return [(ROLL_ACTIONS[f], 1 / 3) for f in (1, 2, 3)]

    def legal_actions(self) -> list[int]:
        return [ROLL_ACTIONS[f] for f in (1, 2, 3)] if self.roll is None else [GO, STEP]

    def apply(self, action: int) -> None:
        if self.roll is None:
            self.roll = action - 1
            self.decisions_left = 2 if self.roll == 3 else 1
            return
        self.pos[self.player] += self.roll if action == GO else 1
        if self.pos[self.player] >= TARGET:
            self._winner = self.player
            return
        self.decisions_left -= 1
        if self.decisions_left == 0:
            self.player = (self.player + 1) % 3
            self.roll = None

    def clone(self) -> "DiceRace":
        g = DiceRace.__new__(DiceRace)
        g.pos = list(self.pos)
        g.player, g.roll, g.decisions_left, g._winner = self.player, self.roll, self.decisions_left, self._winner
        return g

    def is_terminal(self) -> bool:
        return self._winner is not None

    def winner(self) -> int | None:
        return self._winner

    def encode(self) -> np.ndarray:
        rel_pos = [self.pos[(self.player + k) % 3] / TARGET for k in range(3)]
        roll = np.zeros(3)
        if self.roll is not None:
            roll[self.roll - 1] = 1
        return np.array([*rel_pos, *roll, self.decisions_left / 2], dtype=np.float32)
