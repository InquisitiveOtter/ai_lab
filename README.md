# ai_lab

Game-playing AI experiments built on **gamelab**, a small game-agnostic AlphaZero toolkit:
MCTS guided by a policy/value network, trained on parallel self-play.

```
gamelab/        shared library (no game-specific code)
projects/<name> one directory per game
tests/          tests for gamelab, using tic-tac-toe and a tiny dice game
```

## Setup

Python 3.14, Windows. Install a **CUDA build of PyTorch first**, since PyPI's default
Windows wheel is CPU-only. Pick the command for your CUDA version at
<https://pytorch.org/get-started/locally/>, for example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

`torch` is listed in `pyproject.toml` without a version, so `pip install -e .` keeps
whatever build you installed.

## The pieces

| Module | What it provides |
|---|---|
| `gamelab.game` | `Game`: the interface a game implements. `CHANCE` for random events. |
| `gamelab.mcts` | `MCTS` (PUCT, N players, chance nodes, batched across games), `RolloutEvaluator`, `UniformEvaluator` |
| `gamelab.network` | `PolicyValueNet` (conv ResNet or MLP), `NetworkEvaluator`, `get_device`, checkpoints |
| `gamelab.selfplay` | `play_games` (lockstep, batched inference), `SelfPlayPool` (worker processes) |
| `gamelab.replay` | `ReplayBuffer` (ring buffer, save/load) |
| `gamelab.train` | `TrainConfig`, `Trainer`: self-play, train, evaluate, checkpoint, log |
| `gamelab.agents` | `RandomAgent`, `MCTSAgent`, `rollout_mcts_agent` |
| `gamelab.evaluate` | `play_match`, `evaluate_against`: seat-rotating matches with win/draw/loss |
| `gamelab.plotting` | `plot_training`, `compare_runs` from a run's `metrics.jsonl` |

## Adding a game

A `Game` instance is one game state. Implement:

```python
import numpy as np
from gamelab import Game

class MyGame(Game):
    num_players = 2
    action_size = 42                # every action in the game gets an index
    observation_shape = (3, 6, 7)   # (C, H, W) uses a conv net; anything else uses an MLP

    def current_player(self) -> int: ...       # or CHANCE for a dice roll / card draw
    def legal_actions(self) -> list[int]: ...
    def apply(self, action: int) -> None: ...  # mutate in place; step() = clone + apply
    def clone(self) -> "MyGame": ...
    def is_terminal(self) -> bool: ...
    def winner(self) -> int | None: ...        # None for a draw / ongoing
    def encode(self) -> np.ndarray: ...        # float32, from the mover's point of view

    # optional
    def chance_outcomes(self): ...   # [(action, prob), ...] when current_player() == CHANCE
    def returns(self): ...           # per-player outcome in [-1, 1]; default from winner()
    def symmetries(self, obs, policy): ...  # data augmentation for boards
```

Players don't need to alternate (one player can take several actions in a turn), and
any number of players works: values are per-player vectors throughout.

## Training

```python
# projects/mygame/train.py
from gamelab import MCTSConfig, PolicyValueNet, SelfPlayConfig, TrainConfig, Trainer, RandomAgent, rollout_mcts_agent
from projects.mygame.game import MyGame

if __name__ == "__main__":  # required: self-play workers are spawned processes
    config = TrainConfig(
        name="mygame-v1",
        iterations=200,
        games_per_iteration=256,
        selfplay=SelfPlayConfig(mcts=MCTSConfig(num_simulations=200), concurrent_games=128),
        num_workers=0,  # >0 to run self-play in separate processes
    )
    net = PolicyValueNet.for_game(MyGame, width=128, blocks=6)
    baselines = {"random": RandomAgent(0), "mcts200": rollout_mcts_agent(200, seed=0)}
    Trainer(MyGame, net, config, baselines).run()
```

```powershell
python -m projects.mygame.train
```

Each iteration appends a line to `runs/<name>/metrics.jsonl` and saves
`checkpoints/<name>/latest.pt` (plus `iter_XXXXX.pt` every `checkpoint_every`). To resume,
call `trainer.load("checkpoints/<name>/latest.pt")` before `run()`.

**Throughput.** On one GPU, the main lever is `concurrent_games`. That many games run in
lockstep, and each MCTS simulation evaluates one leaf per game in a single batch. Add
`num_workers` when Python game logic, not inference, is the bottleneck. Each worker
then runs its own batch of games, with its own copy of the network on the self-play device.

## Evaluating and plotting

```python
from gamelab import RandomAgent, MCTSAgent, MCTSConfig, NetworkEvaluator, load_checkpoint, play_match
from gamelab.plotting import plot_training

net, _ = load_checkpoint("checkpoints/mygame-v1/latest.pt")
agent = MCTSAgent(NetworkEvaluator(net), MCTSConfig(num_simulations=400, dirichlet_epsilon=0))
print(play_match(MyGame, agent, RandomAgent(), num_games=100))   # e.g. "97W 1D 2L (score 0.975)"

plot_training("runs/mygame-v1", save_to="mygame-v1.png")
```

## Tests

```powershell
python -m pytest              # fast suite, ~10 s
python -m pytest -m slow -s   # also trains tic-tac-toe end to end, ~30 s on CPU
```
