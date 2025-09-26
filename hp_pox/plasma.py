"""Simple plasma surrogate parameterisations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import cantera as ct


@dataclass
class PlasmaSurrogateConfig:
    mode: str = "none"
    start_position_m: float = 0.0
    end_position_m: float | None = None
    plasma_power_W: float = 0.0
    enthalpy_increase_J_per_kg: float | None = None
    exit_temperature_K: float | None = None
    radical_injection: Mapping[str, float] = field(default_factory=dict)
    radical_molar_flow_kmol_per_s: float = 0.0
    injection_width_m: float = 0.05

    def is_active(self, position: float) -> bool:
        if self.mode == "none":
            return False
        end = self.end_position_m if self.end_position_m is not None else self.start_position_m
        if self.mode == "radical" and self.injection_width_m > 0:
            return abs(position - end) <= 0.5 * self.injection_width_m
        return self.start_position_m <= position <= end


@dataclass
class PlasmaSource:
    heat_W_per_m: float = 0.0
    species_kmol_per_m: Dict[str, float] = field(default_factory=dict)


def apply_plasma_surrogates(
    config: PlasmaSurrogateConfig | None, position: float, gas: ct.Solution, omega: ct.ndarray
) -> PlasmaSource:
    if config is None or not config.is_active(position):
        return PlasmaSource()
    if config.mode == "thermal":
        length = (config.end_position_m or config.start_position_m) - config.start_position_m
        length = max(length, 1e-6)
        heat_per_m = config.plasma_power_W / length
        if config.exit_temperature_K is not None:
            cp_mass = gas.cp_mass
            delta_T = max(config.exit_temperature_K - gas.T, 0.0)
            heat_per_m = max(heat_per_m, cp_mass * delta_T * gas.density)
        if config.enthalpy_increase_J_per_kg is not None:
            heat_per_m = max(heat_per_m, config.enthalpy_increase_J_per_kg * gas.density)
        return PlasmaSource(heat_W_per_m=heat_per_m)
    if config.mode == "radical":
        total_flow = config.radical_molar_flow_kmol_per_s
        if total_flow <= 0:
            return PlasmaSource()
        fractions = config.radical_injection
        if not fractions:
            return PlasmaSource()
        norm = sum(max(v, 0.0) for v in fractions.values())
        if norm <= 0:
            return PlasmaSource()
        species_injection = {
            species: total_flow * max(value, 0.0) / norm for species, value in fractions.items()
        }
        width = max(config.injection_width_m, 1e-6)
        for key in species_injection:
            species_injection[key] /= width
        return PlasmaSource(species_kmol_per_m=species_injection)
    return PlasmaSource()
