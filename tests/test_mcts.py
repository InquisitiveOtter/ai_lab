import numpy as np
import pytest

from gamelab import MCTS, MCTSConfig, RolloutEvaluator, UniformEvaluator
from gamelab.mcts import select_action, visit_policy
from tests.dice_race import DiceRace
from tests.tictactoe import TicTacToe


def search(state, sims=400, evaluator=None, seed=0):
    rng = np.random.default_rng(seed)
    mcts = MCTS(evaluator or RolloutEvaluator(rng=rng), MCTSConfig(num_simulations=sims), rng)
    return mcts.search([state])[0]


def best(root):
    return int(root.actions[np.argmax(root.child_n)])


def test_takes_immediate_win():
    assert best(search(TicTacToe.from_string("XX.OO...."))) == 2  # X completes the top row


def test_blocks_opponent_win():
    assert best(search(TicTacToe.from_string("OO.X...X."))) == 2  # X must block


def test_wins_as_second_player():
    # O to move: completing the middle row beats blocking X's top row
    g = TicTacToe.from_string("XX.OO..X.")
    assert g.current_player() == 1
    assert best(search(g)) == 5


def test_finds_win_with_only_terminal_values():
    root = search(TicTacToe.from_string("XX.OO...."), evaluator=UniformEvaluator())
    assert best(root) == 2


def test_visit_counts_and_policy():
    root = search(TicTacToe(), sims=100, evaluator=UniformEvaluator())
    assert root.child_n.sum() == 100
    assert root.visits == 101  # the expansion visit plus one per simulation
    pi = visit_policy(root, 1.0)
    assert pi.shape == (9,) and np.isclose(pi.sum(), 1.0)
    greedy = visit_policy(root, 0.0)
    assert greedy.max() == 1.0 and greedy.sum() == 1.0
    assert select_action(root, 0.0, np.random.default_rng(0)) == int(np.argmax(greedy))


def test_batched_search_matches_independent_trees():
    states = [TicTacToe.from_string("XX.OO...."), TicTacToe.from_string("OO.X...X.")]
    rng = np.random.default_rng(1)
    roots = MCTS(RolloutEvaluator(rng=rng), MCTSConfig(num_simulations=300), rng).search(states)
    assert [best(r) for r in roots] == [2, 2]


def test_root_noise_changes_priors_but_keeps_distribution():
    rng = np.random.default_rng(0)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(num_simulations=10), rng)
    root = mcts.search([TicTacToe()], add_noise=True)[0]
    assert np.isclose(root.priors.sum(), 1.0)
    assert not np.allclose(root.priors, 1 / 9)


def test_chance_nodes_and_three_players():
    g = DiceRace()
    g.pos = [7, 0, 0]
    g = g.step(2)  # rolled a 1: "go" and "step" both win
    root = search(g, sims=50)
    assert root.child_w.shape == (2, 3)
    # player 0 wins from either child; their Q is +1, the others' -1
    assert np.allclose(root.q(), 1.0)

    g = DiceRace()
    g.pos = [6, 0, 0]
    g = g.step(3)  # rolled a 2: "go" wins now, "step" does not
    assert best(search(g, sims=200)) == 0


def test_search_through_chance_nodes_from_the_start():
    g = DiceRace().step(4)  # rolled a 3
    root = search(g, sims=200)
    # chance children exist below decision nodes and are sampled, not evaluated
    child = root.children[int(np.argmax(root.child_n))]
    assert child is not None


def test_search_rejects_chance_root():
    with pytest.raises(ValueError):
        search(DiceRace())
