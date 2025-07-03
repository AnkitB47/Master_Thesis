import numpy as np
from metaheuristics import run_abc, run_bees, run_bnb


def simple_eval(g: np.ndarray) -> float:
    return g.sum()


def test_abc():
    best, hist = run_abc(4, simple_eval)
    assert best.sum() <= 4
    assert len(hist) > 0


def test_bees():
    best, hist = run_bees(4, simple_eval)
    assert best.sum() <= 4
    assert len(hist) > 0


def test_bnb():
    best, hist = run_bnb(3, simple_eval)
    assert best.sum() <= 3
    assert len(hist) > 0
