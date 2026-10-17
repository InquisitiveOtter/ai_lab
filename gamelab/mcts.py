"""PUCT Monte Carlo tree search (AlphaZero style), for N players and chance nodes.

Values are vectors with one entry per player (indexed by absolute player), so
backup needs no sign-flipping and works when the same player moves twice in a
row. At chance nodes the search samples an outcome by its probability.

`MCTS.search` runs several independent trees in lockstep: each simulation picks
one leaf per tree, and all leaves go to the evaluator as a single batch. This is
what keeps a GPU busy during self-play.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from gamelab.game import Game, sample_chance


class Evaluator(Protocol):
    def evaluate(self, states: list[Game]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate non-terminal decision states.

        Returns `(priors, values)`: priors of shape (B, action_size), only read
        at each state's legal actions; values of shape (B, num_players), in
        [-1, 1], indexed by absolute player.
        """
        ...


@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    """Weight of root noise when `search(add_noise=True)`. 0 disables it."""
    fpu: float = 0.0
    """Q-value assumed for unvisited children ("first play urgency")."""


class Node:
    """A state in the tree. Edge statistics live in arrays on the parent."""

    __slots__ = ("state", "player", "actions", "priors", "children", "child_n", "child_w", "visits")

    def __init__(self, state: Game):
        self.state = state
        self.player = None if state.is_terminal() else state.current_player()
        self.actions: np.ndarray | None = None  # set on expansion
        self.priors: np.ndarray | None = None  # chance nodes: outcome probabilities
        self.children: list[Node | None] = []
        self.child_n: np.ndarray | None = None
        self.child_w: np.ndarray | None = None  # (num_children, num_players)
        self.visits = 0

    @property
    def is_terminal(self) -> bool:
        return self.player is None

    @property
    def is_expanded(self) -> bool:
        return self.actions is not None

    def expand(self, actions, priors) -> None:
        k = len(actions)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.priors = np.asarray(priors, dtype=np.float64)
        self.children = [None] * k
        self.child_n = np.zeros(k)
        self.child_w = np.zeros((k, self.state.num_players))

    def child(self, i: int) -> "Node":
        node = self.children[i]
        if node is None:
            node = self.children[i] = Node(self.state.step(int(self.actions[i])))
        return node

    def q(self, fpu: float = 0.0) -> np.ndarray:
        """Mean value of each child for the player to move here."""
        n = self.child_n
        return np.where(n > 0, self.child_w[:, self.player] / np.maximum(n, 1), fpu)


class MCTS:
    def __init__(self, evaluator: Evaluator, config: MCTSConfig | None = None, rng=None):
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

    def search(self, states: list[Game], add_noise: bool = False) -> list[Node]:
        """Search from each state (a player to move, game not over); return the roots."""
        roots = [Node(s) for s in states]
        if any(r.is_terminal or r.state.is_chance() for r in roots):
            raise ValueError("search() needs states where a player is to move")
        self._evaluate_and_expand(roots, [[] for _ in roots])
        if add_noise and self.config.dirichlet_epsilon > 0:
            for root in roots:
                self._add_noise(root)

        for _ in range(self.config.num_simulations):
            pending, paths = [], []
            for root in roots:
                path, leaf = self._select(root)
                if leaf.is_terminal:
                    self._backup(path, leaf, leaf.state.returns())
                else:
                    pending.append(leaf)
                    paths.append(path)
            if pending:
                self._evaluate_and_expand(pending, paths)
        return roots

    def _select(self, node: Node) -> tuple[list[tuple[Node, int]], Node]:
        path = []
        while not node.is_terminal:
            if node.state.is_chance():
                if not node.is_expanded:
                    actions, probs = zip(*node.state.chance_outcomes())
                    node.expand(actions, np.asarray(probs) / np.sum(probs))
                i = int(self.rng.choice(len(node.actions), p=node.priors))
            elif not node.is_expanded:
                break
            else:
                u = self.config.c_puct * node.priors * math.sqrt(node.visits) / (1 + node.child_n)
                i = int(np.argmax(node.q(self.config.fpu) + u))
            path.append((node, i))
            node = node.child(i)
        return path, node

    def _evaluate_and_expand(self, leaves: list[Node], paths) -> None:
        priors, values = self.evaluator.evaluate([leaf.state for leaf in leaves])
        for leaf, path, p, v in zip(leaves, paths, priors, values):
            legal = leaf.state.legal_actions()
            p = np.asarray(p, dtype=np.float64)[legal]
            total = p.sum()
            leaf.expand(legal, p / total if total > 0 else np.full(len(legal), 1 / len(legal)))
            self._backup(path, leaf, v)

    def _add_noise(self, root: Node) -> None:
        eps = self.config.dirichlet_epsilon
        noise = self.rng.dirichlet(np.full(len(root.actions), self.config.dirichlet_alpha))
        root.priors = (1 - eps) * root.priors + eps * noise

    @staticmethod
    def _backup(path, leaf: Node, value) -> None:
        leaf.visits += 1
        for node, i in path:
            node.visits += 1
            node.child_n[i] += 1
            node.child_w[i] += value


def visit_policy(root: Node, temperature: float = 1.0, rng=None) -> np.ndarray:
    """Root visit counts as a distribution over the full action space.

    temperature 0 puts all mass on the most-visited action (ties broken at random).
    """
    counts = root.child_n
    if temperature == 0:
        best = np.flatnonzero(counts == counts.max())
        rng = rng if rng is not None else np.random.default_rng()
        probs = np.zeros(len(counts))
        probs[rng.choice(best)] = 1.0
    else:
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
    policy = np.zeros(root.state.action_size, dtype=np.float32)
    policy[root.actions] = probs
    return policy


def select_action(root: Node, temperature: float, rng: np.random.Generator) -> int:
    policy = visit_policy(root, temperature, rng)
    return int(rng.choice(len(policy), p=policy / policy.sum()))


class UniformEvaluator:
    """Uniform priors and zero values: MCTS guided by nothing but terminal states."""

    def evaluate(self, states):
        n = states[0].action_size
        return np.full((len(states), n), 1.0 / n), np.zeros((len(states), states[0].num_players))


class RolloutEvaluator:
    """Uniform priors; values from random playouts. The classic "pure MCTS" baseline."""

    def __init__(self, num_rollouts: int = 1, max_moves: int | None = None, rng=None):
        self.num_rollouts = num_rollouts
        self.max_moves = max_moves
        self.rng = rng if rng is not None else np.random.default_rng()

    def evaluate(self, states):
        priors, _ = UniformEvaluator().evaluate(states)
        values = np.array(
            [np.mean([self.playout(s) for _ in range(self.num_rollouts)], axis=0) for s in states]
        )
        return priors, values

    def playout(self, state: Game) -> np.ndarray:
        state = state.clone()
        moves = 0
        while not state.is_terminal():
            if self.max_moves is not None and moves >= self.max_moves:
                return np.zeros(state.num_players, dtype=np.float32)
            if state.is_chance():
                state.apply(sample_chance(state, self.rng))
            else:
                state.apply(int(self.rng.choice(state.legal_actions())))
                moves += 1
        return state.returns()
