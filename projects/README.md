# Projects

One directory per game, e.g. `projects/catan/`. A typical project:

```
projects/<name>/
    __init__.py
    game.py      # class MyGame(gamelab.Game)
    net.py       # optional: a custom network (same forward contract as PolicyValueNet)
    train.py     # builds a TrainConfig and runs gamelab.Trainer
    README.md    # rules simplifications, results, notes
```

Run from the repo root: `python -m projects.<name>.train`.
