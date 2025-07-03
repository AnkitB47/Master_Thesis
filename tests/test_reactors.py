import cantera as ct
from reactor.batch import run_constant_volume


def test_constant_volume_runs():
    gas = ct.Solution('gri30.yaml')
    res = run_constant_volume(gas, 1000.0, ct.one_atm, {'CH4':0.5, 'O2':1.0, 'N2':3.76}, 0.001, nsteps=5)
    assert len(res.time) == 5
