import numpy as np

from gamelab import CHANCE, resolve_chance
from tests.dice_race import DiceRace
from tests.tictactoe import TicTacToe


def test_tictactoe_win_and_draw():
    g = TicTacToe.from_string("XX.OO....")
    assert g.current_player() == 0
    assert not g.is_terminal()
    won = g.step(2)
    assert won.is_terminal() and won.winner() == 0
    assert won.returns().tolist() == [1.0, -1.0]
    assert not g.is_terminal(), "step() must not mutate the original"

    draw = TicTacToe.from_string("XOXXOOOXX")
    assert draw.is_terminal() and draw.winner() is None
    assert draw.returns().tolist() == [0.0, 0.0]


def test_tictactoe_encoding_is_from_movers_perspective():
    g = TicTacToe().step(4)  # X in the center, O to move
    obs = g.encode()
    assert obs.shape == TicTacToe.observation_shape
    assert obs[0].sum() == 0 and obs[1, 1, 1] == 1  # O has nothing; opponent plane has X


def test_tictactoe_symmetries_keep_policy_aligned_with_board():
    g = TicTacToe().step(0)  # X in a corner
    policy = np.zeros(9, dtype=np.float32)
    policy[1] = 1.0  # O plays next to it
    for obs, pol in g.symmetries(g.encode(), policy):
        x_cell = int(np.flatnonzero(obs[1].reshape(9))[0])
        target = int(np.argmax(pol))
        # the target cell stays edge-adjacent to the X corner under every symmetry
        assert abs(x_cell // 3 - target // 3) + abs(x_cell % 3 - target % 3) == 1
    assert len({o.tobytes() + p.tobytes() for o, p in g.symmetries(g.encode(), policy)}) == 8


def test_dice_race_chance_and_double_moves():
    g = DiceRace()
    assert g.current_player() == CHANCE
    assert sum(p for _, p in g.chance_outcomes()) == 1.0
    g = g.step(4)  # rolled a 3: two decisions
    assert g.current_player() == 0
    g = g.step(0)
    assert g.current_player() == 0
    g = g.step(0)
    assert g.current_player() == CHANCE and g.player == 1
    assert g.pos == [6, 0, 0]


def test_resolve_chance_reaches_a_decision():
    rng = np.random.default_rng(0)
    g = resolve_chance(DiceRace(), rng)
    assert g.current_player() == 0
