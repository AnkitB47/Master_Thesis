"""Unit tests for plug-flow energy balance and plasma element conservation."""

from __future__ import annotations

import numpy as np
import cantera as ct

from hp_pox.configuration import (
    CaseDefinition,
    GeometryProfile,
    GeometrySegment,
    HeatLossModel,
    InletStream,
)
from hp_pox.pfr import PlugFlowOptions, PlugFlowSolver
from hp_pox.plasma import PlasmaSurrogateConfig, apply_plasma_surrogates


def _build_minimal_case() -> CaseDefinition:
    stream = InletStream(
        name="feed",
        mass_flow_kg_per_h=50.0,
        temperature_K=1100.0,
        composition={"CH4": 0.3, "O2": 0.2, "N2": 0.5},
        basis="mole",
    )
    geometry = GeometryProfile([GeometrySegment(length_m=0.3, diameter_m=0.05)])
    heat_loss = HeatLossModel(mode="adiabatic")
    return CaseDefinition(
        name="unit-test",
        pressure_bar=5.0,
        target_temperature_K=1400.0,
        residence_time_s=0.1,
        friction_factor=0.0,
        streams=[stream],
        geometry=geometry,
        heat_loss=heat_loss,
    )


def test_exothermic_mixture_heats_up():
    case = _build_minimal_case()
    options = PlugFlowOptions(output_points=20, include_wall_heat_loss=False)
    solver = PlugFlowSolver("data/gri30.yaml", case, options=options)
    result = solver.solve()
    assert result.temperature_K[-1] > result.temperature_K[0]


def test_plasma_radical_source_conserves_elements():
    gas = ct.Solution("data/gri30.yaml")
    gas.TPX = 1200.0, 5e5, "CH4:0.4, O2:0.2, N2:0.4"
    omega = np.zeros(gas.n_species)
    config = PlasmaSurrogateConfig(
        mode="radical",
        start_position_m=0.0,
        end_position_m=0.05,
        radical_injection={"H": 0.5, "O": 0.5},
        radical_molar_flow_kmol_per_s=1e-6,
        injection_width_m=0.05,
    )
    source = apply_plasma_surrogates(config, 0.05, gas, omega)
    assert source.species_kmol_per_m, "Expected non-empty radical injection"
    totals = {elem: 0.0 for elem in ("C", "H", "O", "N") if elem in gas.element_names}
    for species, rate in source.species_kmol_per_m.items():
        gas.species_index(species)
        for elem in totals:
            totals[elem] += rate * gas.n_atoms(species, elem)
    assert all(abs(value) < 1e-8 for value in totals.values())
