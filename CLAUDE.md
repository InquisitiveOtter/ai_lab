# ai_lab

Monorepo for game-playing AI experiments (AlphaZero-style). Python 3.14 on Windows; PyTorch with CUDA.

## Layout

- `gamelab/` — **shared, game-agnostic code only.** The `Game` interface, MCTS, networks, self-play, replay buffer, training loop, evaluation, plotting. Nothing game-specific goes here: no game names, rules, encodings or constants. If a project needs a new general feature (e.g. a learning-rate schedule, a new baseline agent), add it here in a game-agnostic form.
- `projects/<name>/` — **one directory per game.** The game's `Game` subclass, any custom network, its training/eval scripts and project notes. Projects import from `gamelab`; they never import from each other.
- `tests/` — tests for `gamelab`, using small test games (`tests/tictactoe.py`, `tests/dice_race.py`). These test games are fixtures, not projects.
- `_reference/` — read-only reference code from earlier work (gitignored). Use it for ideas; don't import from it or copy game-specific code into `gamelab/`.

## Conventions

- Run everything from the repo root: `python -m projects.<name>.train`, `python -m pytest`.
- Self-play workers use `spawn` (Windows), so game classes and network classes must live in importable modules, not in `__main__` or notebooks. Pass the `Game` subclass itself as the game factory.
- Network value outputs are ordered relative to the player to move (index 0 = mover); `Game.encode()` should encode from the mover's perspective. See the docstring in `gamelab/network.py`.
- Randomness in a game (dice, card draws) is modelled as chance nodes: `current_player()` returns `CHANCE` and `chance_outcomes()` lists the outcomes. Don't roll dice inside `apply()`.
- Training outputs go to `runs/<name>/` (metrics, config) and `checkpoints/<name>/`; both are gitignored.
- Don't pin the torch version; it is installed separately with CUDA.

## Testing

- `python -m pytest` runs the fast suite (~10 s).
- `python -m pytest -m slow -s` also runs an end-to-end check that tic-tac-toe is learned (~30 s on CPU).
- Changes to `gamelab/` should keep both passing; add tests with the small test games rather than with a project's game.
