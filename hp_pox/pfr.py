"""Plug-flow solver tailored to the HP-POX benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp

from .compat import reconcile_feed_with_mechanism
from .configuration import CaseDefinition, GeometryProfile, HeatLossModel, InletStream
from .plasma import PlasmaSource, PlasmaSurrogateConfig, apply_plasma_surrogates

R_UNIVERSAL = ct.gas_constant


@dataclass
class PlugFlowOptions:
    method: str = "BDF"
    atol: float = 1e-9
    rtol: float = 1e-6
    max_step: float | None = None
    output_points: int = 200
    ignition_method: str = "dTdx"
    ignition_threshold: float = 150.0  # K/m
    ignition_temperature_K: float = 1100.0
    requested_species: Sequence[str] | None = None
    include_wall_heat_loss: bool = True


@dataclass
class PlugFlowResult:
    positions_m: np.ndarray
    temperature_K: np.ndarray
    pressure_Pa: np.ndarray
    molar_flows: Dict[str, np.ndarray]
    mole_fractions: Dict[str, np.ndarray]
    mass_flow_rate_kg_per_s: np.ndarray
    metrics: Dict[str, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dataframe(self):
        import pandas as pd

        data = {
            "x_m": self.positions_m,
            "T_K": self.temperature_K,
            "P_Pa": self.pressure_Pa,
            "m_dot_kg_per_s": self.mass_flow_rate_kg_per_s,
        }
        for sp, values in self.mole_fractions.items():
            data[f"X_{sp}"] = values
        for sp, values in self.molar_flows.items():
            data[f"F_{sp}_kmol_per_s"] = values
        return pd.DataFrame(data)


class PlugFlowSolver:
    """Thin wrapper around SciPy's IVP integrator for plug-flow reactors."""

    def __init__(
        self,
        mechanism: str | Path | ct.Solution,
        case: CaseDefinition,
        options: PlugFlowOptions | None = None,
        plasma: PlasmaSurrogateConfig | None = None,
        feed_compat_policy: str = "lump_to_propane",
    ) -> None:
        if isinstance(mechanism, ct.Solution):
            self.gas = mechanism
        else:
            self.gas = ct.Solution(str(mechanism))
        self.case = case
        self.options = options or PlugFlowOptions()
        self.plasma = plasma
        self.feed_compat_policy = feed_compat_policy
        self._validate_stream_species()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self) -> PlugFlowResult:
        length = self.case.geometry.total_length
        y0, initial_state = self._initial_state()

        method = self.options.method
        max_step = self.options.max_step or length / (self.options.output_points * 4)
        solution = solve_ivp(
            fun=lambda x, y: self._rhs(x, y),
            t_span=(0.0, length),
            y0=y0,
            method=method,
            atol=self.options.atol,
            rtol=self.options.rtol,
            dense_output=True,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        xs = np.linspace(0.0, length, self.options.output_points)
        states = solution.sol(xs)
        return self._postprocess(xs, states, initial_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initial_state(self) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        gas = self.gas.clone()
        pressure = self.case.pressure_Pa
        total_mass_flow = 0.0
        total_enthalpy_flow = 0.0
        total_molar_flow = np.zeros(gas.n_species)
        for stream in self.case.streams:
            cleaned_composition = reconcile_feed_with_mechanism(
                gas,
                stream.composition,
                policy=self.feed_compat_policy,
            )
            reconciled_stream = replace(stream, composition=cleaned_composition)
            stream_flow, stream_enthalpy, molar_flow = _stream_properties(
                reconciled_stream, gas, pressure
            )
            total_mass_flow += stream_flow
            total_enthalpy_flow += stream_enthalpy
            total_molar_flow += molar_flow
        if total_mass_flow <= 0:
            raise ValueError("Total mass flow rate must be positive")
        mixture_enthalpy = total_enthalpy_flow / total_mass_flow
        total_kmol = total_molar_flow.sum()
        if total_kmol <= 0:
            raise ValueError("Total molar flow must be positive")
        composition = (total_molar_flow / total_kmol).clip(min=0.0)
        gas.HPX = mixture_enthalpy, pressure, composition
        temperature = gas.T
        y0 = np.concatenate(([temperature], total_molar_flow, [pressure]))
        initial = {
            "molar_flow": total_molar_flow,
            "mass_flow": total_mass_flow,
            "composition": composition,
            "temperature": temperature,
            "pressure": pressure,
        }
        return y0, initial

    def _rhs(self, x: float, y: np.ndarray) -> np.ndarray:
        gas = self.gas
        nsp = gas.n_species
        temperature = y[0]
        molar_flow = y[1 : 1 + nsp]
        pressure = y[-1]
        total_kmol = molar_flow.sum()
        if total_kmol <= 0:
            raise ValueError("Total molar flow went non-positive during integration")
        composition = np.clip(molar_flow / total_kmol, 1e-32, None)
        gas.TPX = temperature, pressure, composition
        area = self.case.geometry.area_at(x)
        perimeter = self.case.geometry.perimeter_at(x)
        hydraulic_diameter = self.case.geometry.hydraulic_diameter_at(x)

        omega = gas.net_production_rates  # kmol/m^3/s
        h = gas.partial_molar_enthalpies  # J/kmol
        cp = gas.partial_molar_cp  # J/kmol/K

        plasma_source = apply_plasma_surrogates(self.plasma, x, gas, omega)

        dFdx = area * omega
        if plasma_source.species_kmol_per_m:
            injection = np.zeros_like(molar_flow)
            for species, value in plasma_source.species_kmol_per_m.items():
                try:
                    idx = gas.species_index(species)
                except ValueError:
                    continue
                injection[idx] += value
            dFdx += injection

        cp_flow = float(np.dot(cp, molar_flow))
        if cp_flow == 0.0:
            raise ValueError("Zero heat capacity flow encountered")

        wall_model = self.case.heat_loss
        q_loss = 0.0
        if self.options.include_wall_heat_loss and wall_model.mode != "adiabatic":
            u_value = wall_model.u_value(x)
            tw = wall_model.wall_temperature(x)
            if u_value is not None and tw is not None:
                q_loss = perimeter * u_value * (temperature - tw)

        reaction_term = float(np.dot(omega, h))
        extra_heat = plasma_source.heat_W_per_m
        dTdx = (-area * reaction_term - q_loss + extra_heat) / cp_flow

        molecular_weights = gas.molecular_weights  # kg/kmol
        mass_flow = float(np.dot(molar_flow, molecular_weights))
        rho = gas.density
        velocity = mass_flow / (rho * area)
        friction = self.case.friction_factor
        dPdx = 0.0
        if friction > 0:
            dPdx = -2.0 * friction * rho * velocity * velocity / hydraulic_diameter

        return np.concatenate(([dTdx], dFdx, [dPdx]))

    def _postprocess(
        self,
        xs: np.ndarray,
        states: np.ndarray,
        initial_state: Mapping[str, np.ndarray],
    ) -> PlugFlowResult:
        gas = self.gas
        nsp = gas.n_species
        temperatures = states[0, :]
        molar_flows = states[1 : 1 + nsp, :]
        pressures = states[-1, :]

        species_names = gas.species_names
        mole_fraction_profile: Dict[str, np.ndarray] = {}
        molar_flow_profile: Dict[str, np.ndarray] = {}
        mass_flow_profile = np.zeros_like(xs)
        molecular_weights = gas.molecular_weights

        for idx, name in enumerate(species_names):
            molar_flow_profile[name] = molar_flows[idx, :]

        total_kmol = molar_flows.sum(axis=0)
        mole_fractions = molar_flows / total_kmol
        for idx, name in enumerate(species_names):
            mole_fraction_profile[name] = mole_fractions[idx, :]

        mass_flow_profile = molar_flows.T @ molecular_weights

        metrics = self._compute_metrics(
            xs,
            temperatures,
            pressures,
            species_names,
            molar_flow_profile,
            initial_state,
        )
        metadata = {
            "case": self.case.name,
            "options": self.options,
            "plasma": self.plasma,
        }
        requested_species = self.options.requested_species
        if requested_species is not None:
            mole_fraction_profile = {
                sp: mole_fraction_profile[sp] for sp in requested_species if sp in mole_fraction_profile
            }
            molar_flow_profile = {
                sp: molar_flow_profile[sp] for sp in requested_species if sp in molar_flow_profile
            }
        return PlugFlowResult(
            positions_m=xs,
            temperature_K=temperatures,
            pressure_Pa=pressures,
            molar_flows=molar_flow_profile,
            mole_fractions=mole_fraction_profile,
            mass_flow_rate_kg_per_s=mass_flow_profile,
            metrics=metrics,
            metadata=metadata,
        )

    def _compute_metrics(
        self,
        xs: np.ndarray,
        temperatures: np.ndarray,
        pressures: np.ndarray,
        species_names: Sequence[str],
        molar_flows: Mapping[str, np.ndarray],
        initial_state: Mapping[str, np.ndarray],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        initial_flows = initial_state["molar_flow"]
        species_index = {name: i for i, name in enumerate(species_names)}
        options = self.options

        def species_flow(name: str) -> np.ndarray:
            return molar_flows.get(name, np.zeros_like(xs))

        for species in ("CH4", "CO2"):
            if species in species_index:
                fin = float(initial_flows[species_index[species]])
                fout = float(species_flow(species)[-1])
                if fin > 0:
                    metrics[f"{species}_conversion"] = 1.0 - fout / fin

        if "H2" in molar_flows and "CO" in molar_flows:
            co = species_flow("CO")
            h2 = species_flow("H2")
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = h2 / co
            metrics["H2_CO_ratio"] = float(ratio[-1]) if ratio.size else float("nan")

        metrics["pressure_drop_Pa"] = float(pressures[0] - pressures[-1])

        if options.ignition_method == "dTdx":
            gradients = np.gradient(temperatures, xs)
            idx = np.argmax(np.abs(gradients))
            if abs(gradients[idx]) >= options.ignition_threshold:
                metrics["ignition_position_m"] = float(xs[idx])
        elif options.ignition_method == "temperature":
            above = np.where(temperatures >= options.ignition_temperature_K)[0]
            if above.size:
                metrics["ignition_position_m"] = float(xs[above[0]])

        return metrics

    def _validate_stream_species(self) -> None:
        for stream in self.case.streams:
            try:
                reconcile_feed_with_mechanism(
                    self.gas,
                    stream.composition,
                    policy=self.feed_compat_policy,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Stream '{stream.name}' cannot be reconciled with mechanism {self.gas.name}: {exc}"
                ) from exc


def _stream_properties(
    stream: InletStream, gas: ct.Solution, pressure: float
) -> tuple[float, float, np.ndarray]:
    stream_gas = gas.clone()
    composition = np.array([stream.composition.get(name, 0.0) for name in stream_gas.species_names])
    if stream.basis.lower() == "mole":
        total = composition.sum()
        if total <= 0:
            raise ValueError(f"Stream {stream.name} has zero mole-fraction sum")
        composition /= total
        stream_gas.TPX = stream.temperature_K, pressure, composition
    else:
        mass = composition
        total = mass.sum()
        if total <= 0:
            raise ValueError(f"Stream {stream.name} has zero mass-fraction sum")
        mass /= total
        stream_gas.TPY = stream.temperature_K, pressure, mass
    mass_flow = stream.mass_flow_kg_per_h / 3600.0
    enthalpy_flow = mass_flow * stream_gas.enthalpy_mass
    molar_flow = mass_flow / stream_gas.mean_molecular_weight * stream_gas.X
    return mass_flow, enthalpy_flow, molar_flow
