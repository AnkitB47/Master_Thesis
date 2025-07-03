"""Simple reactor network handling multiple connected reactors."""

from __future__ import annotations

import cantera as ct
import numpy as np
from dataclasses import dataclass
from typing import List

from .batch import BatchResult


@dataclass
class Connection:
    source: int
    target: int
    rate: float  # mass flow rate [kg/s]


def run_network(gas: ct.Solution, reactors: List[ct.Reactor], connections: List[Connection], tf: float, nsteps: int = 1000) -> List[BatchResult]:
    """Integrate a simple network of reactors."""

    sim = ct.ReactorNet(reactors)
    results = [BatchResult(np.zeros(nsteps), np.zeros(nsteps), np.zeros((nsteps, len(gas.species_names)))) for _ in reactors]

    dt = tf / nsteps
    for step in range(nsteps):
        sim.advance(sim.time + dt)
        for conn in connections:
            mdot = conn.rate
            reactors[conn.target].mass += mdot * dt
            reactors[conn.source].mass -= mdot * dt
        for i, r in enumerate(reactors):
            results[i].time[step] = sim.time
            results[i].temperature[step] = r.T
            results[i].mass_fractions[step, :] = r.thermo.Y

    return results
