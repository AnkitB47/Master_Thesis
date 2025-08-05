import cantera as ct
from reactor.batch import run_constant_pressure


def test_ch4_consumed():
    gas = ct.Solution('gri30.yaml')
    Y0 = {'CH4': 0.5, 'O2': 1.0, 'N2': 3.76}
    res = run_constant_pressure(gas, 1000.0, ct.one_atm, Y0, 0.05, nsteps=20)
    idx = gas.species_index('CH4')
    assert res.mass_fractions[-1, idx] < res.mass_fractions[0, idx]
