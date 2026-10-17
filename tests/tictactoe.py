"""Tic-tac-toe: a small, fully solvable game for testing the gamelab core."""

import numpy as np

from gamelab import Game

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


class TicTacToe(Game):
    """Cells 0-8 row by row. Board values: 0 empty, 1 player 0 (X), 2 player 1 (O)."""

    num_players = 2
    action_size = 9
    observation_shape = (3, 3, 3)

    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.player = 0
        self._winner: int | None = None
        self._moves = 0

    def current_player(self) -> int:
        return self.player

    def legal_actions(self) -> list[int]:
        return np.flatnonzero(self.board == 0).tolist()

    def apply(self, action: int) -> None:
        if self.board[action] != 0 or self.is_terminal():
            raise ValueError(f"illegal move {action}")
        self.board[action] = self.player + 1
        self._moves += 1
        if any(self.board[a] == self.board[b] == self.board[c] == self.player + 1 for a, b, c in LINES):
            self._winner = self.player
        self.player = 1 - self.player

    def clone(self) -> "TicTacToe":
        g = TicTacToe.__new__(TicTacToe)
        g.board = self.board.copy()
        g.player = self.player
        g._winner = self._winner
        g._moves = self._moves
        return g

    def is_terminal(self) -> bool:
        return self._winner is not None or self._moves == 9

    def winner(self) -> int | None:
        return self._winner

    def encode(self) -> np.ndarray:
        """Planes: the mover's marks, the opponent's marks, a constant plane for who moves first."""
        grid = self.board.reshape(3, 3)
        return np.stack([
            grid == self.player + 1,
            grid == 2 - self.player,
            np.full((3, 3), self.player == 0),
        ]).astype(np.float32)

    def symmetries(self, observation, policy):
        out = []
        grid = policy.reshape(3, 3)
        for k in range(4):
            obs_k, pol_k = np.rot90(observation, k, axes=(1, 2)), np.rot90(grid, k)
            out.append((obs_k.copy(), pol_k.reshape(9).copy()))
            out.append((np.flip(obs_k, axis=2).copy(), np.fliplr(pol_k).reshape(9).copy()))
        return out

    @classmethod
    def from_string(cls, s: str) -> "TicTacToe":
        """Build a position from 9 characters of 'X', 'O', '.' (row by row); X moves first."""
        g = cls()
        xs = [i for i, ch in enumerate(s) if ch == "X"]
        os_ = [i for i, ch in enumerate(s) if ch == "O"]
        for i in range(len(xs) + len(os_)):
            g.apply(xs[i // 2] if i % 2 == 0 else os_[i // 2])
        return g

    def __str__(self) -> str:
        chars = ".XO"
        return "\n".join("".join(chars[v] for v in self.board[r * 3:r * 3 + 3]) for r in range(3))
