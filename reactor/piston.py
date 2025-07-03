"""Simple piston reactor model."""

from __future__ import annotations

import cantera as ct
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .batch import BatchResult


@dataclass
class PistonOptions:
    a: float = 5e-4
    phi: float = 2 * np.pi
    heat_coeff: float = 0.3e3


def run_piston(
    gas: ct.Solution,
    T0: float,
    P0: float,
    Y0: dict,
    tf: float,
    options: PistonOptions = PistonOptions(),
    nsteps: int = 1000,
) -> BatchResult:
    """Integrate a variable-volume piston reactor."""

    gas.TPY = T0, P0, Y0
    reactor = ct.IdealGasReactor(gas, energy="on")
    sim = ct.ReactorNet([reactor])

    time = []
    T = []
    Y = []

    dt = tf / nsteps
    delay: Optional[float] = None
    last_dTdt = 0.0

    for _ in range(nsteps):
        t = sim.time + dt
        # update volume according to piston motion
        v_dot = -options.a * options.phi * np.sin(options.phi * t)
        reactor.volume += v_dot * dt
        # heat loss
        q = (400.0 - reactor.T) * options.heat_coeff
        reactor.heat_transfer_coeff = q
        sim.advance(t)
        time.append(sim.time)
        T.append(reactor.T)
        Y.append(reactor.Y)
        dTdt = (reactor.T - T[-2]) / dt if len(T) > 1 else 0.0
        if last_dTdt > 0 and dTdt <= 0 and delay is None:
            delay = time[-2]
        last_dTdt = dTdt

    return BatchResult(np.array(time), np.array(T), np.array(Y), ignition_delay=delay)
