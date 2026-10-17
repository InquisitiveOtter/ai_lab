"""Agents that choose moves. Batched (`act_batch`) so many games can share one search."""

from abc import ABC, abstractmethod

import numpy as np

from gamelab.game import Game
from gamelab.mcts import MCTS, Evaluator, MCTSConfig, RolloutEvaluator, select_action


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act_batch(self, states: list[Game]) -> list[int]:
        """One action per state. Every state has a player to move."""

    def act(self, state: Game) -> int:
        return self.act_batch([state])[0]


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act_batch(self, states):
        return [int(self.rng.choice(s.legal_actions())) for s in states]


class MCTSAgent(Agent):
    """Plays the most-visited move (or samples, with temperature > 0) after a search."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        temperature: float = 0.0,
        name: str = "mcts",
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.mcts = MCTS(evaluator, config, self.rng)
        self.temperature = temperature
        self.name = name

    def act_batch(self, states):
        roots = self.mcts.search(states, add_noise=False)
        return [select_action(r, self.temperature, self.rng) for r in roots]


def rollout_mcts_agent(num_simulations: int = 200, max_moves: int | None = None, seed: int | None = None) -> MCTSAgent:
    """Pure MCTS with random playouts: a stronger baseline than random play, no learning involved."""
    rng = np.random.default_rng(seed)
    return MCTSAgent(
        RolloutEvaluator(max_moves=max_moves, rng=rng),
        MCTSConfig(num_simulations=num_simulations, dirichlet_epsilon=0.0),
        name=f"rollout_mcts{num_simulations}",
        seed=seed,
    )
