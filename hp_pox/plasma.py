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
        end = config.end_position_m if config.end_position_m is not None else config.start_position_m
        length = max(end - config.start_position_m, 1e-6)
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
        balanced = _enforce_element_conservation(species_injection, gas)
        return PlasmaSource(species_kmol_per_m=balanced)
    return PlasmaSource()


def _enforce_element_conservation(
    injection: Mapping[str, float], gas: ct.Solution
) -> Dict[str, float]:
    """Return a species source profile that conserves C/H/O/N elements."""

    adjusted: Dict[str, float] = dict(injection)
    if not injection:
        return adjusted

    element_balance = {elem: 0.0 for elem in ("C", "H", "O", "N") if elem in gas.element_names}
    if not element_balance:
        return adjusted

    for species, rate in injection.items():
        try:
            gas.species_index(species)
        except ValueError as exc:
            raise ValueError(f"Plasma injection species '{species}' missing from mechanism") from exc
        for elem in element_balance:
            element_balance[elem] += rate * gas.n_atoms(species, elem)

    def apply_removal(target_species: str, atoms: Dict[str, float], rate: float) -> None:
        if target_species not in gas.species_names:
            raise ValueError(
                f"Cannot conserve elements: species '{target_species}' missing from mechanism"
            )
        adjusted[target_species] = adjusted.get(target_species, 0.0) - rate
        for elem, count in atoms.items():
            if elem in element_balance:
                element_balance[elem] -= rate * count

    if "C" in element_balance and element_balance["C"] > 0:
        if "CH4" not in gas.species_names:
            raise ValueError("CH4 required to balance carbon for radical injection")
        atoms = {elem: gas.n_atoms("CH4", elem) for elem in element_balance}
        removal = element_balance["C"] / max(atoms.get("C", 0.0), 1e-12)
        apply_removal("CH4", atoms, removal)

    if "O" in element_balance and element_balance["O"] > 1e-12:
        if "O2" not in gas.species_names:
            raise ValueError("O2 required to balance oxygen for radical injection")
        atoms = {elem: gas.n_atoms("O2", elem) for elem in element_balance}
        removal = element_balance["O"] / max(atoms.get("O", 0.0), 1e-12)
        apply_removal("O2", atoms, removal)

    if "H" in element_balance and element_balance["H"] > 1e-12:
        target = "H2" if "H2" in gas.species_names else None
        if target is None:
            raise ValueError("H2 required to balance hydrogen for radical injection")
        atoms = {elem: gas.n_atoms(target, elem) for elem in element_balance}
        removal = element_balance["H"] / max(atoms.get("H", 0.0), 1e-12)
        apply_removal(target, atoms, removal)

    if "N" in element_balance and element_balance["N"] > 1e-12:
        if "N2" not in gas.species_names:
            raise ValueError("N2 required to balance nitrogen for radical injection")
        atoms = {elem: gas.n_atoms("N2", elem) for elem in element_balance}
        removal = element_balance["N"] / max(atoms.get("N", 0.0), 1e-12)
        apply_removal("N2", atoms, removal)

    residual = max(abs(value) for value in element_balance.values()) if element_balance else 0.0
    if residual > 1e-8:
        raise ValueError(f"Failed to conserve elements during radical injection (residual={residual:.2e})")
    return adjusted
