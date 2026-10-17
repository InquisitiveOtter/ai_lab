import numpy as np

from gamelab import MCTSConfig, PolicyValueNet, SelfPlayConfig, SelfPlayPool, UniformEvaluator, play_games
from tests.dice_race import DiceRace
from tests.tictactoe import TicTacToe

FAST = SelfPlayConfig(mcts=MCTSConfig(num_simulations=8), concurrent_games=4)


def test_records_have_consistent_targets():
    records = play_games(TicTacToe, UniformEvaluator(), 6, FAST, np.random.default_rng(0))
    assert len(records) == 6
    for r in records:
        n = r.length * 8  # 8 symmetries per position
        assert r.observations.shape == (n, 3, 3, 3)
        assert r.policies.shape == (n, 9) and np.allclose(r.policies.sum(axis=1), 1.0)
        assert r.values.shape == (n, 2)
        # first sample is X to move; value target is X's result
        assert r.values[0, 0] == r.returns[0]
        if r.winner is not None:
            # with relative values, the mover's entry alternates between moves
            assert r.values[0, 0] == -r.values[8, 0]


def test_three_player_chance_game():
    records = play_games(DiceRace, UniformEvaluator(), 3, FAST, np.random.default_rng(0))
    for r in records:
        assert r.winner is not None and r.returns[r.winner] == 1
        assert r.values.shape == (r.length, 3)
        assert np.all(r.policies[:, 2:] == 0)  # chance actions are never policy targets


def test_max_moves_truncates_as_draw():
    cfg = SelfPlayConfig(mcts=MCTSConfig(num_simulations=4), max_moves=3)
    (r,) = play_games(TicTacToe, UniformEvaluator(), 1, cfg, np.random.default_rng(0))
    assert r.truncated and r.length == 3 and r.winner is None
    assert np.all(r.values == 0)


def test_pool_inline_and_with_workers():
    net = PolicyValueNet.for_game(TicTacToe, width=16, blocks=1)
    with SelfPlayPool(TicTacToe, net, FAST, num_workers=0, device="cpu") as pool:
        assert len(pool.play(3, net)) == 3
    with SelfPlayPool(TicTacToe, net, FAST, num_workers=2, device="cpu") as pool:
        records = pool.play(5, net, seed=1)
        assert len(records) == 5
        assert len(pool.play(2, net, seed=2)) == 2  # weights re-sent, workers reused
