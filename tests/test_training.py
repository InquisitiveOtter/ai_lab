import json

import pytest

from gamelab import (
    MCTSConfig, PolicyValueNet, RandomAgent, SelfPlayConfig, TrainConfig, Trainer, rollout_mcts_agent,
)
from gamelab.evaluate import play_match
from gamelab.plotting import load_metrics, plot_training
from tests.tictactoe import TicTacToe


def tiny_config(tmp_path, **kw):
    defaults = dict(
        name="t", iterations=2, games_per_iteration=4,
        selfplay=SelfPlayConfig(mcts=MCTSConfig(num_simulations=8), concurrent_games=4),
        min_buffer_size=1, batch_size=16, train_steps_per_iteration=3,
        eval_every=2, eval_games=4, checkpoint_every=1, device="cpu",
        runs_dir=str(tmp_path / "runs"), checkpoints_dir=str(tmp_path / "checkpoints"),
    )
    return TrainConfig(**{**defaults, **kw})


def test_loop_writes_metrics_checkpoints_and_plots(tmp_path):
    net = PolicyValueNet.for_game(TicTacToe, width=16, blocks=1)
    trainer = Trainer(TicTacToe, net, tiny_config(tmp_path))
    trainer.run()

    metrics = load_metrics(tmp_path / "runs" / "t")
    assert [m["iteration"] for m in metrics] == [1, 2]
    assert "train/loss" in metrics[0]
    assert "eval/random/score" in metrics[1] and "eval/random/score" not in metrics[0]
    assert json.loads((tmp_path / "runs" / "t" / "config.json").read_text())["iterations"] == 2
    assert (tmp_path / "checkpoints" / "t" / "iter_00002.pt").exists()

    plot_training(tmp_path / "runs" / "t", save_to=tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0

    # resume: continues from the saved iteration
    resumed = Trainer(TicTacToe, PolicyValueNet.for_game(TicTacToe, width=16, blocks=1), tiny_config(tmp_path, iterations=3))
    resumed.load(tmp_path / "checkpoints" / "t" / "latest.pt")
    assert resumed.iteration == 2
    resumed.run()
    assert [m["iteration"] for m in load_metrics(tmp_path / "runs" / "t")] == [1, 2, 3]


def test_loop_with_worker_processes(tmp_path):
    net = PolicyValueNet.for_game(TicTacToe, width=16, blocks=1)
    Trainer(TicTacToe, net, tiny_config(tmp_path, num_workers=2, eval_every=0)).run()
    assert len(load_metrics(tmp_path / "runs" / "t")) == 2


@pytest.mark.slow
def test_learns_tictactoe(tmp_path):
    """End to end: a short run should learn to never lose to random play and to
    hold its own against pure MCTS. Takes about 30 seconds on CPU."""
    cfg = tiny_config(
        tmp_path, iterations=15, games_per_iteration=64,
        selfplay=SelfPlayConfig(mcts=MCTSConfig(num_simulations=32), concurrent_games=64, temperature_moves=4),
        min_buffer_size=500, batch_size=128, train_steps_per_iteration=100, eval_every=0, lr=3e-3,
    )
    trainer = Trainer(TicTacToe, PolicyValueNet.for_game(TicTacToe, width=32, blocks=2), cfg)
    trainer.run()
    agent = trainer.agent()
    vs_random = play_match(TicTacToe, agent, RandomAgent(0), 50)
    vs_mcts = play_match(TicTacToe, agent, rollout_mcts_agent(100, seed=0), 20)
    print(f"vs random: {vs_random}; vs rollout MCTS: {vs_mcts}")
    assert vs_random.losses == 0 and vs_random.wins >= 35
    assert vs_mcts.score >= 0.5
