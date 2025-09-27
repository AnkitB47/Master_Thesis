"""Plug-flow solver tailored to the HP-POX benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import copy
import logging
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
        self.logger = logging.getLogger(__name__)
        tracked = []
        for symbol in ("C", "H", "O", "N"):
            try:
                self.gas.element_index(symbol)
            except ValueError:
                continue
            tracked.append(symbol)
        self._tracked_elements = tracked
        if tracked:
            self._element_matrix = np.array(
                [
                    [self.gas.n_atoms(species, elem) for species in self.gas.species_names]
                    for elem in tracked
                ],
                dtype=float,
            )
        else:
            self._element_matrix = np.zeros((0, self.gas.n_species))
        self._initial_element_totals: np.ndarray | None = None
        self._validate_stream_species()
        self._diagnostics: Dict[str, object] = {
            "max_abs_omega": 0.0,
            "reaction_term": [],
            "wall_term": [],
            "plasma_term": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self) -> PlugFlowResult:
        length = self.case.geometry.total_length
        y0, initial_state = self._initial_state()

        method = self.options.method
        max_step = self.options.max_step
        if max_step is None:
            max_step = length / (self.options.output_points * 12)
        first_step = max(length / (self.options.output_points * 10), 1e-9)

        def temperature_event(x, y):
            return 4000.0 - y[0]

        temperature_event.terminal = True
        temperature_event.direction = -1

        def extinction_event(x, y):
            total = float(np.sum(np.maximum(y[1 : 1 + self.gas.n_species], 0.0)))
            return total

        extinction_event.terminal = True
        extinction_event.direction = -1

        solution = solve_ivp(
            fun=lambda x, y: self._rhs(x, y),
            t_span=(0.0, length),
            y0=y0,
            method=method,
            atol=self.options.atol,
            rtol=self.options.rtol,
            dense_output=True,
            max_step=max_step,
            first_step=first_step,
            events=[temperature_event, extinction_event],
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
        try:
            gas = ct.Solution(self.gas.source)
            gas.TPX = self.gas.T, self.gas.P, self.gas.X
        except Exception:
            gas = copy.deepcopy(self.gas)
        pressure = self.case.pressure_Pa
        total_mass_flow = 0.0
        total_enthalpy_flow = 0.0
        total_molar_flow = np.zeros(gas.n_species)
        species_names = gas.species_names
        reconciliation_logs: list[str] = []
        for stream in self.case.streams:
            cleaned_composition, mapping = reconcile_feed_with_mechanism(
                gas,
                stream.composition,
                policy=self.feed_compat_policy,
                return_mapping=True,
            )
            if mapping:
                formatted = ", ".join(f"{missing}->{sink}" for missing, sink in mapping.items())
                reconciliation_logs.append(f"{stream.name}: {formatted}")
            reconciled_stream = replace(stream, composition=cleaned_composition)
            stream_flow, stream_enthalpy, molar_flow = _stream_properties(
                reconciled_stream, gas, pressure
            )
            total_mass_flow += stream_flow
            total_enthalpy_flow += stream_enthalpy
            total_molar_flow += molar_flow
        if reconciliation_logs:
            self.logger.info("Feed reconciliation applied: %s", "; ".join(reconciliation_logs))
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
        if self._element_matrix.size:
            self._initial_element_totals = self._element_matrix @ total_molar_flow
            inventory = {
                elem: float(self._initial_element_totals[i]) for i, elem in enumerate(self._tracked_elements)
            }
            self.logger.debug("Inlet element inventory (kmol): %s", inventory)
        composition_preview = {
            species_names[i]: float(value)
            for i, value in enumerate(composition)
            if value > 1e-6
        }
        self.logger.debug(
            "Inlet mixture: T=%.2f K, P=%.2f Pa, X=%s",
            temperature,
            pressure,
            {k: round(v, 6) for k, v in list(composition_preview.items())[:8]},
        )
        self.logger.debug(
            "Initial molar flow (kmol/s): %s",
            {
                species_names[i]: float(total_molar_flow[i])
                for i in range(len(species_names))
                if total_molar_flow[i] > 0
            },
        )
        self.logger.debug("Initial mass flow: %.6e kg/s", total_mass_flow)
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
        temperature = float(y[0])
        molar_flow_raw = y[1 : 1 + nsp]
        pressure = float(y[-1])

        # use a tiny floor only for building composition (do NOT mutate the ODE state)
        molar_flow_pos = np.maximum(molar_flow_raw, 1e-40)
        total_kmol = float(molar_flow_pos.sum())
        if not np.isfinite(total_kmol) or total_kmol <= 1e-40:
            # bail out gracefully (let integrator backtrack)
            return np.concatenate(([0.0], np.zeros(nsp), [0.0]))

        composition = np.clip(molar_flow_pos / total_kmol, 1e-32, 1.0)
        gas.TPX = temperature, pressure, composition
        molar_flow = molar_flow_raw  # keep original for derivatives

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

        if self._element_matrix.size:
            current_elements = self._element_matrix @ molar_flow_raw
            residual = current_elements - self._initial_element_totals
            drift = float(np.max(np.abs(residual)))
            if drift > 1e-8:
                details = {
                    elem: float(residual[i]) for i, elem in enumerate(self._tracked_elements)
                }
                raise ValueError(
                    f"Elemental imbalance detected at x={x:.4f} m (max drift={drift:.3e}): {details}"
                )

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

        molecular_weights = gas.molecular_weights  # kg/kmol
        mass_flow = float(np.dot(molar_flow, molecular_weights))
        rho = gas.density
        velocity = mass_flow / (rho * area)
        cp_mass = gas.cp_mass
        if velocity <= 0.0:
            raise ValueError("Non-positive axial velocity encountered")
        if cp_mass <= 0.0:
            raise ValueError("Non-positive heat capacity encountered")

        reaction_term = float(np.dot(omega, h))  # J/m^3/s
        extra_heat = plasma_source.heat_W_per_m  # W/m
        reaction_source = 0.0
        wall_source = 0.0
        plasma_source_term = 0.0
        if rho * cp_mass <= 0:
            raise ValueError("Non-positive density*cp term encountered")
        reaction_source = -(reaction_term) / (rho * cp_mass)
        if mass_flow > 0:
            wall_source = -q_loss / (mass_flow * cp_mass)
            plasma_source_term = extra_heat / (mass_flow * cp_mass)
        dTdx = (reaction_source + wall_source + plasma_source_term) / velocity
        self._diagnostics["reaction_term"].append(reaction_source)
        self._diagnostics["wall_term"].append(wall_source)
        self._diagnostics["plasma_term"].append(plasma_source_term)
        self._diagnostics["max_abs_omega"] = max(
            float(self._diagnostics["max_abs_omega"]), float(np.max(np.abs(omega)))
        )

        friction = self.case.friction_factor
        dPdx = 0.0
        if friction > 0:
            dPdx = -friction * rho * velocity * velocity / (2.0 * hydraulic_diameter)

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
        self._log_diagnostics()
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

        ignition_position = None
        if options.ignition_method == "dTdx":
            ignition_position = _detect_ignition_by_gradient(
                xs,
                temperatures,
                threshold=options.ignition_threshold,
            )
        elif options.ignition_method == "temperature":
            ignition_position = _detect_ignition_by_temperature(
                xs, temperatures, options.ignition_temperature_K
            )
        if ignition_position is not None:
            metrics["ignition_position_m"] = ignition_position

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

    def _log_diagnostics(self) -> None:
        if not self._diagnostics:
            return
        self.logger.debug(
            "Max |omega| observed: %.3e kmol/m^3/s",
            float(self._diagnostics.get("max_abs_omega", 0.0)),
        )
        for key in ("reaction_term", "wall_term", "plasma_term"):
            series = self._diagnostics.get(key, [])
            if not series:
                continue
            array = np.asarray(series, dtype=float)
            self.logger.debug(
                "dT/dx contribution %s: mean=%.3e, min=%.3e, max=%.3e",
                key,
                float(np.mean(array)),
                float(np.min(array)),
                float(np.max(array)),
            )
        self._diagnostics = {
            "max_abs_omega": 0.0,
            "reaction_term": [],
            "wall_term": [],
            "plasma_term": [],
        }


def _detect_ignition_by_temperature(xs: np.ndarray, temperatures: np.ndarray, threshold: float) -> float | None:
    above = np.where(temperatures >= threshold)[0]
    if above.size:
        return float(xs[above[0]])
    return None


def _detect_ignition_by_gradient(
    xs: np.ndarray,
    temperatures: np.ndarray,
    threshold: float,
) -> float | None:
    gradients = np.gradient(temperatures, xs)
    candidates = np.where(np.abs(gradients) >= threshold)[0]
    if not candidates.size:
        return None
    second = np.gradient(gradients, xs)
    for idx in candidates:
        left = second[idx - 1] if idx > 0 else second[idx]
        right = second[idx + 1] if idx + 1 < second.size else second[idx]
        if np.sign(left) == 0 or np.sign(right) == 0:
            return float(xs[idx])
        if np.sign(left) != np.sign(right):
            return float(xs[idx])
    return float(xs[candidates[0]])


def _stream_properties(
    stream: InletStream, gas: ct.Solution, pressure: float
) -> tuple[float, float, np.ndarray]:
    try:
        # Reload the mechanism with same thermo/kinetics
        stream_gas = ct.Solution(gas.source)
    except Exception:
        stream_gas = copy.deepcopy(gas)
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
