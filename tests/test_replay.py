import numpy as np

from gamelab import ReplayBuffer


def make(n, start=0):
    obs = np.arange(start, start + n, dtype=np.float32)[:, None].repeat(4, axis=1)
    return obs, np.ones((n, 3), np.float32) / 3, np.zeros((n, 2), np.float32)


def test_fills_and_wraps():
    buf = ReplayBuffer(10, (4,), 3, 2)
    buf.add(*make(6))
    assert len(buf) == 6
    buf.add(*make(6, start=6))
    assert len(buf) == 10 and buf.total_added == 12
    assert set(buf.observations[:, 0]) == set(range(2, 12))  # the two oldest were overwritten


def test_oversized_batch_keeps_newest():
    buf = ReplayBuffer(5, (4,), 3, 2)
    buf.add(*make(12))
    assert set(buf.observations[:, 0]) == set(range(7, 12))


def test_sample_shapes():
    buf = ReplayBuffer(10, (4,), 3, 2)
    buf.add(*make(7))
    obs, pol, val = buf.sample(32, np.random.default_rng(0))
    assert obs.shape == (32, 4) and pol.shape == (32, 3) and val.shape == (32, 2)
    assert obs[:, 0].max() < 7


def test_save_load_round_trip(tmp_path):
    buf = ReplayBuffer(10, (4,), 3, 2)
    buf.add(*make(13))
    buf.save(tmp_path / "buf.npz")

    same = ReplayBuffer.load(tmp_path / "buf.npz")
    assert len(same) == 10 and same.total_added == 13
    assert set(same.observations[:, 0]) == set(range(3, 13))

    smaller = ReplayBuffer.load(tmp_path / "buf.npz", capacity=4)
    assert set(smaller.observations[:, 0]) == set(range(9, 13))  # newest kept
