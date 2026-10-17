import numpy as np
import torch

from gamelab import NetworkEvaluator, PolicyValueNet, load_checkpoint, save_checkpoint
from gamelab.network import to_absolute, to_relative
from tests.dice_race import DiceRace
from tests.tictactoe import TicTacToe


def test_conv_and_mlp_shapes():
    conv = PolicyValueNet.for_game(TicTacToe, width=16, blocks=1)
    logits, values = conv(torch.zeros(5, *TicTacToe.observation_shape))
    assert logits.shape == (5, 9) and values.shape == (5, 2)

    mlp = PolicyValueNet.for_game(DiceRace, width=16, blocks=1)
    logits, values = mlp(torch.zeros(5, *DiceRace.observation_shape))
    assert logits.shape == (5, 5) and values.shape == (5, 3)
    assert values.abs().max() <= 1


def test_value_rotation_round_trips():
    v = np.array([0.1, 0.2, 0.3])
    assert to_relative(v, 1).tolist() == [0.2, 0.3, 0.1]  # mover first
    for p in range(3):
        assert np.allclose(to_absolute(to_relative(v, p), p), v)


def test_evaluator_masks_illegal_actions_and_uses_absolute_values():
    net = PolicyValueNet.for_game(TicTacToe, width=16, blocks=1)
    states = [TicTacToe(), TicTacToe.from_string("XX.OO....").step(8)]  # second: O to move
    priors, values = NetworkEvaluator(net, "cpu").evaluate(states)
    assert priors.shape == (2, 9) and values.shape == (2, 2)
    assert np.allclose(priors.sum(axis=1), 1.0)
    assert np.all(priors[1][[0, 1, 3, 4, 8]] == 0)  # occupied cells get no prior

    # the same network output, seen from the other player, swaps value order
    _, rel = net.eval()(torch.from_numpy(states[1].encode()[None]))
    assert np.allclose(values[1], rel.detach().numpy()[0][::-1], atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    net = PolicyValueNet.for_game(TicTacToe, width=16, blocks=1)
    opt = torch.optim.AdamW(net.parameters())
    save_checkpoint(tmp_path / "c.pt", net, opt, iteration=7)
    loaded, data = load_checkpoint(tmp_path / "c.pt")  # architecture rebuilt from net_config
    assert data["iteration"] == 7
    x = torch.randn(2, *TicTacToe.observation_shape)
    net.eval(), loaded.eval()
    assert all(torch.allclose(a, b) for a, b in zip(net(x), loaded(x)))
