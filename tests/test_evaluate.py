import numpy as np

from gamelab import RandomAgent, play_match, rollout_mcts_agent
from gamelab.evaluate import play_games
from tests.dice_race import DiceRace
from tests.tictactoe import TicTacToe


def test_random_vs_random_accounts_for_every_game():
    r = play_match(TicTacToe, RandomAgent(0), RandomAgent(1), 50, np.random.default_rng(0))
    assert r.games == 50
    assert 0.3 < r.score < 0.8  # moving first is an advantage; seats rotate


def test_rollout_mcts_beats_random():
    r = play_match(TicTacToe, rollout_mcts_agent(100, seed=0), RandomAgent(0), 20, np.random.default_rng(0))
    assert r.losses == 0 and r.wins >= 15


def test_seats_rotate_and_multiplayer_works():
    finals = play_games(DiceRace, [[RandomAgent(i)] * 3 for i in range(6)], np.random.default_rng(0))
    assert all(f.is_terminal() for f in finals)
    r = play_match(DiceRace, rollout_mcts_agent(30, seed=0), RandomAgent(0), 30, np.random.default_rng(0))
    assert r.games == 30 and r.draws == 0
    assert r.wins / r.games > 1 / 3  # better than a random seat in a 3-player race
