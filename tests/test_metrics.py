from metrics import l2_distance, pv_error
import numpy as np


def test_l2_distance():
    Y1 = np.zeros((3,2))
    Y2 = np.ones((3,2))
    t = np.array([0.0, 0.5, 1.0])
    d = l2_distance(Y1, Y2, t)
    assert d > 0


def test_pv_error_zero():
    pv1 = np.array([0.1, 0.2, 0.3])
    pv2 = pv1.copy()
    assert pv_error(pv1, pv2) < 1e-12
