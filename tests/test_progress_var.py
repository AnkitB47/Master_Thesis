from progress_variable import progress_variable, optimise_weights
import numpy as np


def test_optimise_weights():
    Y = np.random.rand(5, 3)
    pv_ref = progress_variable(Y, [1, 0, 1])
    w = optimise_weights(Y, pv_ref)
    assert len(w) == 3
