import json
import numpy as np
import cantera as ct
from reactor.batch import run_constant_pressure
from progress_variable import progress_variable
import pytest


def test_progress_variable_changes():
    gas = ct.Solution('gri30.yaml')
    Y0 = {'CH4': 0.5, 'O2': 1.0, 'N2': 3.76}
    res = run_constant_pressure(gas, 1000.0, ct.one_atm, Y0, 0.05, nsteps=20)
    with open('data/species_weights.json') as f:
        weights_map = json.load(f)
    species = list(weights_map.keys())
    idxs = [gas.species_index(s) for s in species]
    Y = res.mass_fractions[:, idxs]
    weights = [weights_map[s] for s in species]
    pv = progress_variable(Y, weights)
    assert not np.allclose(pv, pv[0])
    assert pv[-1] > pv[0]


def test_progress_variable_zero_weights():
    Y = np.ones((5, 3))
    weights = [0.0, 0.0, 0.0]
    pv = progress_variable(Y, weights)
    assert np.allclose(pv, 0.0)


def test_progress_variable_weight_mismatch():
    Y = np.zeros((4, 2))
    weights = [1.0]
    with pytest.raises(ValueError):
        progress_variable(Y, weights)
