import cantera as ct
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

@dataclass
class BatchResult:
    """Result of a batch reactor simulation."""

    time: np.ndarray
    temperature: np.ndarray
    mass_fractions: np.ndarray
    ignition_delay: Optional[float] = None


def run_constant_pressure(gas: ct.Solution, T0: float, P0: float, Y0: dict, tf: float, nsteps: int = 1000) -> BatchResult:
    gas.TPY = T0, P0, Y0
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    time = []
    T = []
    Y = []
    dt = tf/nsteps
    for _ in range(nsteps):
        sim.advance(sim.time + dt)
        time.append(sim.time)
        T.append(reactor.T)
        Y.append(reactor.Y)
    return BatchResult(np.array(time), np.array(T), np.array(Y))


def run_constant_volume(
    gas: ct.Solution,
    T0: float,
    P0: float,
    Y0: dict,
    tf: float,
    nsteps: int = 1000,
    detect_ignition: bool = True,
) -> BatchResult:
    """Integrate a constant-volume adiabatic reactor."""

    gas.TPY = T0, P0, Y0
    reactor = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([reactor])
    time = []
    T = []
    Y = []
    delay = None
    last_dTdt = 0.0
    dt = tf / nsteps
    for _ in range(nsteps):
        sim.advance(sim.time + dt)
        time.append(sim.time)
        T.append(reactor.T)
        Y.append(reactor.Y)
        if detect_ignition:
            dTdt = (reactor.T - T[-2]) / dt if len(T) > 1 else 0.0
            if last_dTdt > 0 and dTdt <= 0 and delay is None:
                delay = time[-2]
            last_dTdt = dTdt

    return BatchResult(np.array(time), np.array(T), np.array(Y), ignition_delay=delay)
