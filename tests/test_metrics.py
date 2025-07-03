from metrics import l2_distance
import numpy as np


def test_l2_distance():
    Y1 = np.zeros((3,2))
    Y2 = np.ones((3,2))
    t = np.array([0.0, 0.5, 1.0])
    d = l2_distance(Y1, Y2, t)
    assert d > 0
