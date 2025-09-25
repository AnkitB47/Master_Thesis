import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_pipeline_runs(tmp_path):
    from testing.pipeline import full_pipeline
    out = tmp_path / "out"
    full_pipeline("data/gri30.yaml", str(out), steps=10, tf=0.05)
    assert (out / "ga_fitness.csv").exists()


def test_zero_d_regression(tmp_path):
    import cantera as ct
    import numpy as np
    from mechanism.loader import Mechanism
    from mechanism.mix import methane_air_mole_fractions, mole_to_mass_fractions
    from reactor.batch import run_constant_pressure
    from progress_variable import pv_error_aligned
    from metrics import ignition_delay

    mech = Mechanism('data/gri30.yaml')
    solution = mech.solution
    X0 = methane_air_mole_fractions(1.0)
    Y0 = mole_to_mass_fractions(solution, X0)
    tf = 2.0e-3
    steps = 60
    full = run_constant_pressure(solution, 1500.0, ct.one_atm, Y0, tf, nsteps=steps, use_mole=False)
    red = run_constant_pressure(solution, 1500.0, ct.one_atm, Y0, tf, nsteps=steps, use_mole=False)

    weights = np.ones(len(mech.species_names))
    pv_err = pv_error_aligned(full.mass_fractions, red.mass_fractions, mech.species_names, mech.species_names, weights)
    delay_full, _ = ignition_delay(full.time, full.temperature)
    delay_red, _ = ignition_delay(red.time, red.temperature)

    assert pv_err < 1e-10
    assert abs(delay_red - delay_full) <= 1e-10 + 1e-10 * delay_full
