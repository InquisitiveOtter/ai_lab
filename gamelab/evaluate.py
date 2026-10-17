"""Playing agents against each other and scoring the results."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from gamelab.agents import Agent
from gamelab.game import Game, resolve_chance


@dataclass
class MatchResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """(wins + draws / 2) / games."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def __str__(self) -> str:
        return f"{self.wins}W {self.draws}D {self.losses}L (score {self.score:.3f})"


def play_games(
    game_factory: Callable[[], Game],
    seatings: Sequence[Sequence[Agent]],
    rng: np.random.Generator | None = None,
    max_moves: int | None = None,
) -> list[Game]:
    """Play one game per seating (`seating[p]` plays as player p), all in lockstep.

    At each step, every agent gets all the states where it is to move as one
    batch. Returns the final states; games cut off by `max_moves` are returned
    unfinished.
    """
    rng = rng if rng is not None else np.random.default_rng()
    states = [resolve_chance(game_factory(), rng) for _ in seatings]
    moves = [0] * len(states)
    active = [i for i, s in enumerate(states) if not s.is_terminal()]
    while active:
        by_agent: dict[int, tuple[Agent, list[int]]] = {}
        for i in active:
            agent = seatings[i][states[i].current_player()]
            by_agent.setdefault(id(agent), (agent, []))[1].append(i)
        for agent, idx in by_agent.values():
            for i, action in zip(idx, agent.act_batch([states[i] for i in idx])):
                states[i] = resolve_chance(states[i].step(action), rng)
                moves[i] += 1
        active = [
            i for i in active
            if not states[i].is_terminal() and (max_moves is None or moves[i] < max_moves)
        ]
    return states


def play_match(
    game_factory: Callable[[], Game],
    agent: Agent,
    opponent: Agent,
    num_games: int,
    rng: np.random.Generator | None = None,
    max_moves: int | None = None,
) -> MatchResult:
    """`agent` against `opponent`, rotating `agent` through every seat.

    With more than two players, every seat not taken by `agent` is played by
    `opponent`. A game with no winner counts as a draw.
    """
    num_players = game_factory().num_players
    seats = [g % num_players for g in range(num_games)]
    seatings = [[agent if p == seat else opponent for p in range(num_players)] for seat in seats]
    result = MatchResult()
    for seat, final in zip(seats, play_games(game_factory, seatings, rng, max_moves)):
        winner = final.winner() if final.is_terminal() else None
        if winner is None:
            result.draws += 1
        elif winner == seat:
            result.wins += 1
        else:
            result.losses += 1
    return result


def evaluate_against(
    game_factory: Callable[[], Game],
    agent: Agent,
    baselines: dict[str, Agent],
    num_games: int,
    rng: np.random.Generator | None = None,
    max_moves: int | None = None,
) -> dict[str, MatchResult]:
    return {
        name: play_match(game_factory, agent, opponent, num_games, rng, max_moves)
        for name, opponent in baselines.items()
    }
