"""One-dimensional plug-flow reactor utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PFRConfig:
    """Configuration for an ideal-gas tubular reactor."""

    length: float  # reactor length [m]
    diameter: float  # internal diameter [m]
    mass_flow: float  # total mass flow [kg/s]
    pressure: float  # operating pressure [Pa]
    n_points: int = 400  # sampling points for reporting
    heat_transfer_coeff: float = 0.0  # overall U [W/(m^2 K)]
    wall_temperature: float | Callable[[float], float] | None = None
    enable_heat_loss: bool = False
    max_residence_time: Optional[float] = None
    plasma_length: float = 0.0
    plasma_temperature: Optional[float] = None

    # internal fields initialised later
    _area: float = field(init=False, repr=False)
    _perimeter: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")
        self._area = 0.25 * math.pi * self.diameter**2
        self._perimeter = math.pi * self.diameter

    @property
    def area(self) -> float:
        return self._area

    @property
    def perimeter(self) -> float:
        return self._perimeter


def _as_callable_tw(tw: float | Callable[[float], float] | None) -> Callable[[float], float]:
    if callable(tw):
        return tw  # type: ignore[return-value]
    if tw is None:
        return lambda _x: 300.0
    return lambda _x, val=float(tw): val


def _mass_fractions_array(gas: ct.Solution, mapping: Dict[str, float]) -> np.ndarray:
    vec = np.zeros(gas.n_species, dtype=float)
    for i, name in enumerate(gas.species_names):
        vec[i] = float(mapping.get(name, 0.0))
    s = float(vec.sum())
    if s <= 0:
        raise ValueError("Mass fractions must sum to a positive value")
    return vec / s


def _mole_to_mass(gas: ct.Solution, X: Dict[str, float]) -> Dict[str, float]:
    gas.TPX = 300.0, ct.one_atm, X
    return {sp: float(y) for sp, y in zip(gas.species_names, gas.Y) if y > 0.0}


def _mass_to_mole(Y: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
    denom = np.sum(Y / molecular_weights[None, :], axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return (Y / molecular_weights[None, :]) / denom


@dataclass
class PFRResult:
    """Result of a PFR integration."""

    residence_time: np.ndarray
    x: np.ndarray
    temperature: np.ndarray
    mass_fractions: np.ndarray
    species_names: Iterable[str]
    molecular_weights: np.ndarray
    initial_mass_fractions: np.ndarray
    initial_mole_fractions: np.ndarray
    dTdx: np.ndarray
    ignition_index: int
    ch4_conversion: np.ndarray
    co2_conversion: np.ndarray
    h2_co_ratio: np.ndarray

    @property
    def time(self) -> np.ndarray:
        return self.residence_time

    @property
    def ignition_position(self) -> float:
        return float(self.x[self.ignition_index])

    def resample(self, tau_samples: np.ndarray) -> "PFRResult":
        tau = np.asarray(tau_samples, dtype=float)
        if tau.ndim != 1:
            raise ValueError("tau_samples must be 1-D")
        # clamp to available window
        t0, t1 = float(self.residence_time[0]), float(self.residence_time[-1])
        tau = np.clip(tau, t0, t1)

        # pathological time grids fallback: nearest sampling
        if not np.all(np.diff(self.residence_time) > 0):
            idx = np.searchsorted(self.residence_time, tau, side="left")
            idx = np.clip(idx, 0, len(self.residence_time) - 1)
            x = self.x[idx]
            T = self.temperature[idx]
            Y = self.mass_fractions[idx]
            return _postprocess_result(
                tau,
                x,
                T,
                Y,
                list(self.species_names),
                self.molecular_weights,
                self.initial_mass_fractions,
                self.initial_mole_fractions,
            )

        x = np.interp(tau, self.residence_time, self.x)
        # enforce monotonic x to stabilize gradient
        x = np.maximum.accumulate(x)
        if x.size:
            span = float(x[-1] - x[0]) if x.size > 1 else 1.0
            denom = max(x.size - 1, 1)
            min_step = max(1e-9, abs(span) * 1e-6 / denom)
            ramp = min_step * np.arange(x.size)
            x = np.maximum.accumulate(x + ramp)
        T = np.interp(tau, self.residence_time, self.temperature)
        Y = np.zeros((len(tau), self.mass_fractions.shape[1]))
        for j in range(self.mass_fractions.shape[1]):
            Y[:, j] = np.interp(tau, self.residence_time, self.mass_fractions[:, j])
        return _postprocess_result(
            tau,
            x,
            T,
            Y,
            list(self.species_names),
            self.molecular_weights,
            self.initial_mass_fractions,
            self.initial_mole_fractions,
        )


def _postprocess_result(
    tau: np.ndarray,
    x: np.ndarray,
    temperature: np.ndarray,
    mass_fractions: np.ndarray,
    species_names: Iterable[str],
    molecular_weights: np.ndarray,
    initial_mass_fractions: np.ndarray,
    initial_mole_fractions: np.ndarray,
) -> PFRResult:
    names = list(species_names)

    # --- enforce strictly increasing x & drop bad points
    x = np.asarray(x, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    mass_fractions = np.asarray(mass_fractions, dtype=float)

    mask = np.isfinite(x) & np.isfinite(temperature)
    if mass_fractions.ndim == 2:
        mask &= np.isfinite(mass_fractions).all(axis=1)
    x = x[mask]
    temperature = temperature[mask]
    mass_fractions = mass_fractions[mask]
    tau = np.asarray(tau, dtype=float)[mask]

    if x.size >= 2:
        x = np.maximum.accumulate(x)
        span = float(x[-1] - x[0]) if x.size > 1 else 1.0
        denom = max(x.size - 1, 1)
        min_step = max(1e-9, abs(span) * 1e-6 / denom)
        ramp = min_step * np.arange(x.size)
        x = np.maximum.accumulate(x + ramp)
        keep = np.concatenate(([True], np.diff(x) > 1e-12))
        x = x[keep]
        temperature = temperature[keep]
        mass_fractions = mass_fractions[keep]
        tau = tau[keep]

    # safe gradient
    if len(x) >= 3 and (x[-1] - x[0]) > 0:
        try:
            dTdx = np.gradient(temperature, x, edge_order=2)
        except Exception:
            dTdx = np.gradient(temperature, edge_order=2)
    else:
        dTdx = np.zeros_like(x)

    ign_idx = int(np.argmax(dTdx)) if len(dTdx) else 0

    X = _mass_to_mole(mass_fractions, molecular_weights)
    idx = {s: i for i, s in enumerate(names)}

    ch4_conv = np.zeros(len(tau))
    ch4_in = initial_mole_fractions[idx["CH4"]] if "CH4" in idx else 0.0
    if ch4_in > 0:
        ch4_conv = 1.0 - X[:, idx["CH4"]] / max(ch4_in, 1e-12)

    co2_conv = np.zeros(len(tau))
    if "CO2" in idx and initial_mole_fractions[idx["CO2"]] > 0:
        co2_conv = 1.0 - X[:, idx["CO2"]] / max(initial_mole_fractions[idx["CO2"]], 1e-12)

    h2_co = np.zeros(len(tau))
    if "H2" in idx and "CO" in idx:
        h2_co = X[:, idx["H2"]] / np.clip(X[:, idx["CO"]], 1e-30, None)

    return PFRResult(
        residence_time=tau,
        x=x,
        temperature=temperature,
        mass_fractions=mass_fractions,
        species_names=names,
        molecular_weights=molecular_weights,
        initial_mass_fractions=initial_mass_fractions,
        initial_mole_fractions=initial_mole_fractions,
        dTdx=dTdx,
        ignition_index=ign_idx,
        ch4_conversion=ch4_conv,
        co2_conversion=co2_conv,
        h2_co_ratio=h2_co,
    )


def run_pfr(
    gas: ct.Solution,
    config: PFRConfig,
    T0: float,
    p0: float,
    Y0: Dict[str, float] | np.ndarray,
    *,
    n_points: Optional[int] = None,
) -> PFRResult:
    """Integrate the plug-flow reactor and return axial profiles."""

    if isinstance(Y0, dict):
        Y_init = _mass_fractions_array(gas, Y0)
    else:
        Y_init = np.asarray(Y0, dtype=float)
    gas.TPY = T0, p0, Y_init
    molecular_weights = gas.molecular_weights.copy()
    initial_mass = gas.Y.copy()
    initial_mole = gas.X.copy()

    tw_fun = _as_callable_tw(config.wall_temperature)
    enable_heat_loss = config.enable_heat_loss and config.heat_transfer_coeff > 0.0

    plasma_T0 = T0
    p_oper = p0

    def rhs(tau: float, state: np.ndarray) -> np.ndarray:
        T = float(state[0])
        Y = state[1:-1]
        x = float(state[-1])

        # --- safe normalisation & state clamping
        Y = np.clip(Y, 0.0, None)
        s = Y.sum()
        if not np.isfinite(s) or s <= 1e-20:
            Y = initial_mass.copy()
            s = Y.sum()
        Y = Y / s
        T = float(np.clip(T, 200.0, 4000.0))

        gas.TPY = T, p_oper, Y
        rho = max(gas.density, 1e-6)   # avoid nonpositive density
        cp = max(gas.cp_mass, 1e-3)

        # chemistry
        wdot = gas.net_production_rates  # kmol/m^3/s
        heat_release = -np.dot(gas.partial_molar_enthalpies, wdot)  # -Σ h ω̇
        dTdtau = heat_release / (rho * cp)

        # heat loss to wall
        if enable_heat_loss:
            dTdtau -= (
                config.heat_transfer_coeff * config.perimeter / (rho * config.area * cp)
            ) * (T - tw_fun(x))

        u = config.mass_flow / (rho * config.area)

        # smooth plasma heating (logistic ramp across plasma zone)
        if config.plasma_length > 0 and config.plasma_temperature is not None:
            Lp = max(config.plasma_length, 1e-6)
            xmid = 0.5 * Lp
            k = 12.0 / Lp
            frac = 1.0 / (1.0 + np.exp(-k * (x - xmid)))
            T_tar = plasma_T0 + (config.plasma_temperature - plasma_T0) * np.clip(frac, 0.0, 1.0)
            dTdx_plasma = np.clip(T_tar - T, -800.0, 800.0) / max(Lp / 2.0, 1e-6)
            dTdtau += dTdx_plasma * u

        dYdtau = (wdot * gas.molecular_weights) / rho
        dxdtau = u

        # cap total heating/cooling rate to avoid numeric blow-up
        dTdtau = float(np.clip(dTdtau, -5e6, 5e6))

        return np.concatenate(([dTdtau], dYdtau, [dxdtau]))

    state0 = np.concatenate(([T0], Y_init, [0.0]))
    rho0 = max(gas.density, 1e-6)
    u0 = config.mass_flow / (rho0 * config.area)
    tau_guess = config.length / max(u0, 1e-12)
    t_end = config.max_residence_time or (5.0 * tau_guess)

    event = lambda tau, y: y[-1] - config.length
    event.terminal = True  # type: ignore[attr-defined]
    event.direction = 1  # type: ignore[attr-defined]

    points = int(n_points or config.n_points)
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        state0,
        method="BDF",
        atol=1e-12,           # relaxed from 1e-18
        rtol=1e-6,            # relaxed from 1e-8
        events=event,
        dense_output=False,
        max_step=min(t_end / max(points, 200), 5e-3),  # hard cap
    )

    if sol.y.shape[1] < 2:
        raise RuntimeError("PFR integration failed: insufficient points")

    tau = sol.t
    x = sol.y[-1]
    T = sol.y[0]
    Y = sol.y[1:-1].T

    if points and len(tau) > points:
        tau_grid = np.linspace(tau[0], tau[-1], points)
        return _postprocess_result(
            tau_grid,
            np.interp(tau_grid, tau, x),
            np.interp(tau_grid, tau, T),
            np.column_stack([np.interp(tau_grid, tau, Y[:, j]) for j in range(Y.shape[1])]),
            gas.species_names,
            molecular_weights,
            initial_mass,
            initial_mole,
        )

    return _postprocess_result(
        tau,
        x,
        T,
        Y,
        gas.species_names,
        molecular_weights,
        initial_mass,
        initial_mole,
    )


class PFRRunner:
    """Callable wrapper compatible with batch reactor runners."""

    def __init__(self, config: PFRConfig, base_points: int = 400):
        self.config = config
        self.base_points = base_points

    def __call__(
        self,
        gas: ct.Solution,
        T0: float,
        p0: float,
        X_or_Y0: Dict[str, float],
        tf: float,
        nsteps: int = 200,
        *,
        use_mole: bool = False,
        log_times: bool = False,
        time_grid: Optional[np.ndarray] = None,
    ) -> PFRResult:
        if use_mole:
            Y0 = _mole_to_mass(gas, X_or_Y0)
        else:
            Y0 = X_or_Y0
        points = max(self.base_points, nsteps + 1)
        res = run_pfr(
            gas,
            self.config,
            T0,
            p0,
            Y0,
            n_points=points,
        )
        if time_grid is not None:
            return res.resample(np.asarray(time_grid, dtype=float))
        if log_times:
            tau_grid = np.geomspace(max(res.time[1], 1e-12), res.time[-1], nsteps)
            tau_grid = np.insert(tau_grid, 0, 0.0)
            return res.resample(tau_grid)
        if nsteps and len(res.time) != nsteps + 1:
            tau_grid = np.linspace(res.time[0], res.time[-1], nsteps + 1)
            return res.resample(tau_grid)
        return res
