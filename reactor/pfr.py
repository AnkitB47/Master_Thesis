"""One-dimensional plug-flow reactor utilities."""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Literal, Optional

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse
from scipy.integrate._ivp import bdf as _bdf
from scipy.integrate._ivp import common as _common
from scipy.integrate._ivp import radau as _radau


logger = logging.getLogger(__name__)


class IntegrationBailout(RuntimeError):
    """Raised when the stiff PFR integration exceeds safety limits."""

    def __init__(self, message: str, reason: str, diagnostics: dict[str, float | int | str]):
        super().__init__(message)
        self.reason = reason
        self.diagnostics = diagnostics


@dataclass
class IntegrationMonitor:
    """Light-weight monitor shared between the RHS and solver internals."""

    start_time: float
    timeout: float
    reject_limit: int
    accepted: int = 0
    rejected: int = 0
    num_jac_time: float = 0.0
    max_abs_dTdtau: float = 0.0
    last_tau: float = 0.0
    history_tau: list[float] = field(default_factory=list)
    history_state: list[np.ndarray] = field(default_factory=list)
    termination_reason: str | None = None

    def check_timeout(self) -> None:
        if self.timeout <= 0:
            return
        if time.perf_counter() - self.start_time > self.timeout:
            self.termination_reason = "timeout"
            raise IntegrationBailout(
                "PFR integration aborted due to timeout",
                "timeout",
                self.snapshot(),
            )

    def note_rejection(self) -> None:
        self.rejected += 1
        if self.reject_limit and self.rejected > self.reject_limit:
            self.termination_reason = "too_many_rejections"
            raise IntegrationBailout(
                "PFR integration aborted after excessive step rejections",
                "reject_limit",
                self.snapshot(),
            )

    def note_accept(self, tau: float, state: np.ndarray) -> None:
        self.accepted += 1
        self.last_tau = tau
        self.history_tau.append(float(tau))
        self.history_state.append(np.array(state, copy=True))

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "num_jac_time": self.num_jac_time,
            "max_abs_dTdtau": self.max_abs_dTdtau,
            "timeout_s": self.timeout,
        }


class _InstrumentedRadau(_radau.Radau):
    """Radau integrator that reports step statistics to an ``IntegrationMonitor``."""

    def __init__(self, *args, monitor: IntegrationMonitor | None = None, **kwargs):
        self._pfr_monitor = monitor
        super().__init__(*args, **kwargs)

    def _step_impl(self):  # type: ignore[override]
        monitor = self._pfr_monitor
        if monitor is not None:
            monitor.check_timeout()
        t = self.t
        y = self.y
        f = self.f

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        J = self.J
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if monitor is not None:
                monitor.check_timeout()
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Z0 = np.zeros((3, y.shape[0]))
            else:
                Z0 = self.sol(t + h * _radau.C).T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if monitor is not None:
                    monitor.check_timeout()
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(_radau.MU_REAL / h * self.I - J)
                    LU_complex = self.lu(_radau.MU_COMPLEX / h * self.I - J)

                converged, n_iter, Z, rate = _radau.solve_collocation_system(
                    self.fun,
                    t,
                    y,
                    h,
                    Z0,
                    scale,
                    self.newton_tol,
                    LU_real,
                    LU_complex,
                    self.solve_lu,
                )

                if not converged:
                    if current_jac:
                        break

                    J = self.jac(t, y, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                if monitor is not None:
                    monitor.note_rejection()
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = y + Z[-1]
            ZE = Z.T.dot(_radau.E) / h
            error = self.solve_lu(LU_real, f + ZE)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = _radau.norm(error / scale)
            safety = 0.9 * (2 * _radau.NEWTON_MAXITER + 1) / (
                2 * _radau.NEWTON_MAXITER + n_iter
            )

            if rejected and error_norm > 1:
                error = self.solve_lu(LU_real, self.fun(t, y + error) + ZE)
                error_norm = _radau.norm(error / scale)

            if error_norm > 1:
                if monitor is not None:
                    monitor.note_rejection()
                factor = _radau.predict_factor(
                    h_abs, h_abs_old, error_norm, error_norm_old
                )
                h_abs *= max(_radau.MIN_FACTOR, safety * factor)

                LU_real = None
                LU_complex = None
                rejected = True
            else:
                step_accepted = True

        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = _radau.predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        factor = min(_radau.MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new

        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        self.sol = self._compute_dense_output()

        if monitor is not None:
            monitor.note_accept(self.t, self.y)

        return True, message


class _InstrumentedBDF(_bdf.BDF):
    """BDF integrator that reports step statistics to an ``IntegrationMonitor``."""

    def __init__(self, *args, monitor: IntegrationMonitor | None = None, **kwargs):
        self._pfr_monitor = monitor
        super().__init__(*args, **kwargs)

    def _step_impl(self):  # type: ignore[override]
        monitor = self._pfr_monitor
        if monitor is not None:
            monitor.check_timeout()
        t = self.t
        D = self.D

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            _bdf.change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            _bdf.change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        J = self.J
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        while not step_accepted:
            if monitor is not None:
                monitor.check_timeout()
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                _bdf.change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[: order + 1], axis=0)

            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1 : order + 1].T, gamma[1 : order + 1]) / alpha[order]

            converged = False
            c = h / alpha[order]
            while not converged:
                if monitor is not None:
                    monitor.check_timeout()
                if LU is None:
                    LU = self.lu(self.I - c * J)

                converged, n_iter, y_new, d = _bdf.solve_bdf_system(
                    self.fun,
                    t_new,
                    y_predict,
                    c,
                    psi,
                    LU,
                    self.solve_lu,
                    scale,
                    self.newton_tol,
                )

                if not converged:
                    if current_jac:
                        break
                    J = self.jac(t_new, y_predict)
                    LU = None
                    current_jac = True

            if not converged:
                if monitor is not None:
                    monitor.note_rejection()
                factor = 0.5
                h_abs *= factor
                _bdf.change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = 0.9 * (2 * _bdf.NEWTON_MAXITER + 1) / (
                2 * _bdf.NEWTON_MAXITER + n_iter
            )

            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            error_norm = _bdf.norm(error / scale)

            if error_norm > 1:
                if monitor is not None:
                    monitor.note_rejection()
                factor = max(
                    _bdf.MIN_FACTOR, safety * error_norm ** (-1 / (order + 1))
                )
                h_abs *= factor
                _bdf.change_D(D, order, factor)
                self.n_equal_steps = 0
            else:
                step_accepted = True

        self.n_equal_steps += 1

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.J = J
        self.LU = LU

        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if monitor is not None:
            monitor.note_accept(self.t, self.y)

        if self.n_equal_steps < order + 1:
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = _bdf.norm(error_m / scale)
        else:
            error_m_norm = np.inf

        if order < _bdf.MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = _bdf.norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide="ignore"):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(_bdf.MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        _bdf.change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None

        return True, None


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
    solver: Literal["Radau", "BDF"] = "Radau"
    rtol: float = 1e-6
    atol: float = 1e-12
    first_step: float | None = 1e-7
    max_step_cap: float | None = 5e-4
    jac_sparsity: bool = True
    timeout: float | None = None
    reject_limit: int | None = None

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


def _pfr_jac_sparsity(n_species: int) -> sparse.csr_matrix:
    """Return a conservative sparsity pattern for the PFR Jacobian."""

    n_state = n_species + 2  # temperature + species + axial position
    rows = []
    cols = []

    # Dense temperature row (coupled to all states)
    rows.extend([0] * n_state)
    cols.extend(range(n_state))

    # Species block as tri-diagonal (each species depends on itself and neighbours)
    for i in range(n_species):
        row = i + 1
        for col in (i, i + 1, i + 2):
            if 0 <= col < n_state:
                rows.append(row)
                cols.append(col)

    # Axial position row depends weakly on density/temperature (safe dense row)
    rows.extend([n_state - 1] * n_state)
    cols.extend(range(n_state))

    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_state, n_state))


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
    diagnostics: Dict[str, float | int | str] | None = None

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
                self.diagnostics,
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
            self.diagnostics,
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
    diagnostics: Dict[str, float | int | str] | None = None,
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

    if len(x) >= 2:
        # ensure strictly increasing x before gradient
        x = np.asarray(x, dtype=float)
        eps = 1e-12
        x_mono = x.copy()
        for i in range(1, len(x_mono)):
            if x_mono[i] <= x_mono[i - 1]:
                x_mono[i] = x_mono[i - 1] + eps

        # robust gradient: avoid zero dx near ignition kinks
        dT = np.gradient(temperature)
        dx = np.empty_like(x_mono)
        if len(x_mono) > 2:
            dx[1:-1] = 0.5 * (x_mono[2:] - x_mono[:-2])
        if len(x_mono) >= 2:
            dx[0] = x_mono[1] - x_mono[0]
            dx[-1] = x_mono[-1] - x_mono[-2]
        dx = np.where(np.abs(dx) < eps, np.sign(dx) * eps + (dx == 0) * eps, dx)
        dTdx = dT / dx
        x = x_mono
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
        diagnostics=diagnostics,
    )


def run_pfr(
    gas: ct.Solution,
    config: PFRConfig,
    T0: float,
    p0: float,
    Y0: Dict[str, float] | np.ndarray,
    *,
    n_points: Optional[int] = None,
    mode: Literal["reference", "ga_fast"] = "reference",
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

    timeout = (
        config.timeout
        if config.timeout is not None
        else (8.0 if mode == "ga_fast" else 0.0)
    )
    reject_limit = (
        config.reject_limit
        if config.reject_limit is not None
        else (2000 if mode == "ga_fast" else 0)
    )
    monitor = IntegrationMonitor(
        start_time=time.perf_counter(),
        timeout=max(float(timeout), 0.0),
        reject_limit=max(int(reject_limit), 0),
    )

    state0 = np.concatenate(([T0], Y_init, [0.0]))
    monitor.history_tau.append(0.0)
    monitor.history_state.append(state0.copy())

    def rhs(tau: float, state: np.ndarray) -> np.ndarray:
        if monitor is not None:
            monitor.check_timeout()
        T = float(state[0])
        Y = state[1:-1]
        x = float(state[-1])

        # --- safe normalisation & state clamping
        Y = np.clip(Y, 0.0, None)
        s = float(Y.sum())
        if not np.isfinite(s) or s <= 1e-18:
            logger.debug(
                "Mass fractions degenerated at tau=%.3e (sum=%.3e); resetting to inlet",
                tau,
                s,
            )
            Y = initial_mass.copy()
            s = float(Y.sum())
        if s <= 0.0:
            raise IntegrationBailout(
                "Invalid mass-fraction normalisation",
                "mass_fraction",
                monitor.snapshot(),
            )
        Y = Y / s
        T = float(np.clip(T, 200.0, 4000.0))

        gas.TPY = T, p_oper, Y
        rho = float(gas.density)
        if not np.isfinite(rho) or rho <= 0.0:
            raise IntegrationBailout(
                f"Non-positive density at tau={tau:.3e}: rho={rho}",
                "density",
                monitor.snapshot(),
            )
        cp = max(float(gas.cp_mass), 1e-3)

        # chemistry
        wdot = gas.net_production_rates  # kmol/m^3/s
        heat_release = -float(np.dot(gas.partial_molar_enthalpies, wdot))  # -Σ h ω̇
        dTdtau = heat_release / (rho * cp)

        # heat loss to wall
        if enable_heat_loss:
            dTdtau -= (
                config.heat_transfer_coeff * config.perimeter / (rho * config.area * cp)
            ) * (T - tw_fun(x))

        u = config.mass_flow / (rho * config.area)
        if not np.isfinite(u) or u <= 0.0:
            raise IntegrationBailout(
                f"Non-positive axial velocity at tau={tau:.3e}: u={u}",
                "velocity",
                monitor.snapshot(),
            )

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
        if monitor is not None:
            monitor.max_abs_dTdtau = max(monitor.max_abs_dTdtau, abs(dTdtau))

        return np.concatenate(([dTdtau], dYdtau, [dxdtau]))

    rho0 = max(float(gas.density), 1e-6)
    u0 = config.mass_flow / (rho0 * config.area)
    tau_guess = config.length / max(u0, 1e-12)
    t_end = config.max_residence_time or (5.0 * tau_guess)

    event = lambda tau, y: y[-1] - config.length
    event.terminal = True  # type: ignore[attr-defined]
    event.direction = 1  # type: ignore[attr-defined]

    points = int(n_points or config.n_points)
    base_geom = t_end / max(points, 200)
    if mode == "ga_fast":
        cap = 5e-4 if config.max_step_cap is None else min(config.max_step_cap, 5e-4)
        max_step = min(base_geom, cap)
        rtol = 1e-5
        atol = 1e-9
        solver_choice: str | type[_radau.Radau] | type[_bdf.BDF] = _InstrumentedBDF
    else:
        geom_cap = min(base_geom, config.max_step_cap) if config.max_step_cap is not None else base_geom
        max_step = geom_cap
        rtol = float(config.rtol)
        atol = float(config.atol)
        if config.solver == "BDF":
            solver_choice = _InstrumentedBDF
        else:
            solver_choice = _InstrumentedRadau

    solver_kwargs = {
        "method": solver_choice,
        "rtol": rtol,
        "atol": atol,
        "events": event,
        "dense_output": False,
        "max_step": max(max_step, 1e-12),
        "monitor": monitor,
    }

    if config.first_step is not None:
        solver_kwargs["first_step"] = float(max(config.first_step, 1e-12))
    if config.jac_sparsity:
        solver_kwargs["jac_sparsity"] = _pfr_jac_sparsity(len(Y_init))

    orig_num_jac = _common.num_jac

    def _timed_num_jac(*args, **kwargs):
        start = time.perf_counter()
        try:
            return orig_num_jac(*args, **kwargs)
        finally:
            monitor.num_jac_time += time.perf_counter() - start

    _common.num_jac = _timed_num_jac  # type: ignore[assignment]

    diagnostics: Dict[str, float | int | str] = {}

    try:
        sol = solve_ivp(
            rhs,
            (0.0, t_end),
            state0,
            **solver_kwargs,
        )
    except IntegrationBailout as exc:
        diagnostics = {"status": exc.reason, **exc.diagnostics}
        diagnostics["mode"] = mode
        diagnostics.setdefault("accepted", monitor.accepted)
        diagnostics.setdefault("rejected", monitor.rejected)
        diagnostics.setdefault("num_jac_time", monitor.num_jac_time)
        diagnostics.setdefault("max_abs_dTdtau", monitor.max_abs_dTdtau)
        logger.warning(
            "PFR bailout (%s): %s", exc.reason, exc,
        )
        tau_hist = np.asarray(monitor.history_tau, dtype=float)
        state_hist = np.asarray(monitor.history_state, dtype=float)
        if tau_hist.size < 2:
            tau_hist = np.array([0.0, min(t_end, max(1e-9, base_geom))])
            state_hist = np.vstack([state0, state0])
        tau = tau_hist
        x = state_hist[:, -1]
        T = state_hist[:, 0]
        Y = state_hist[:, 1:-1]
    else:
        diagnostics = {
            "status": "ok" if sol.success else "failed",
            "accepted": monitor.accepted,
            "rejected": monitor.rejected,
            "num_jac_time": monitor.num_jac_time,
            "max_abs_dTdtau": monitor.max_abs_dTdtau,
            "mode": mode,
        }
        tau = sol.t
        if tau.size == 0 or sol.y.shape[1] < 2:
            raise RuntimeError("PFR integration failed: insufficient points")
        x = sol.y[-1]
        T = sol.y[0]
        Y = sol.y[1:-1].T
        if monitor.history_tau and monitor.history_state:
            diagnostics["tau_end_monitor"] = monitor.history_tau[-1]
    finally:
        _common.num_jac = orig_num_jac  # type: ignore[assignment]

    if np.any(np.diff(x) <= 0):
        logger.warning(
            "Non-monotone axial coordinate detected; rebuilding surrogate grid",
        )
        dx = np.diff(x)
        positive = np.where(dx > 0, dx, 0.0)
        total = float(positive.sum())
        if total <= 0 or not np.isfinite(total):
            x_sur = np.linspace(0.0, config.length, len(x))
        else:
            eps = max(total * 1e-6, 1e-12)
            positive = np.where(positive <= eps, eps, positive)
            x_sur = np.empty_like(x)
            x_sur[0] = max(x[0], 0.0)
            x_sur[1:] = x_sur[0] + np.cumsum(positive)
            final = float(x_sur[-1])
            if np.isfinite(final) and final > 0 and config.length > 0:
                x_sur *= config.length / final
        tau_target = (
            np.linspace(tau[0], tau[-1], len(x_sur)) if len(tau) > 1 else tau.copy()
        )
        T = np.interp(tau_target, tau, T)
        Y_interp = np.zeros((len(tau_target), Y.shape[1]))
        for j in range(Y.shape[1]):
            Y_interp[:, j] = np.interp(tau_target, tau, Y[:, j])
        tau = tau_target
        x = x_sur
        Y = Y_interp

    if points and len(tau) > points:
        tau_grid = np.linspace(tau[0], tau[-1], points)
        x_interp = np.interp(tau_grid, tau, x)
        T_interp = np.interp(tau_grid, tau, T)
        Y_interp = np.column_stack(
            [np.interp(tau_grid, tau, Y[:, j]) for j in range(Y.shape[1])]
        )
        res = _postprocess_result(
            tau_grid,
            x_interp,
            T_interp,
            Y_interp,
            gas.species_names,
            molecular_weights,
            initial_mass,
            initial_mole,
            diagnostics,
        )
    else:
        res = _postprocess_result(
            tau,
            x,
            T,
            Y,
            gas.species_names,
            molecular_weights,
            initial_mass,
            initial_mole,
            diagnostics,
        )

    logger.info(
        "PFR run mode=%s status=%s accepted=%d rejected=%d max|dT/dτ|=%.2e jac_time=%.2fs",
        mode,
        res.diagnostics.get("status") if res.diagnostics else "",
        monitor.accepted,
        monitor.rejected,
        monitor.max_abs_dTdtau,
        monitor.num_jac_time,
    )

    return res


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
        integrator_mode: Literal["reference", "ga_fast"] = "reference",
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
            mode=integrator_mode,
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
