"""One-dimensional flame solver wrappers using Cantera."""

from __future__ import annotations

import cantera as ct
import numpy as np
from dataclasses import dataclass


@dataclass
class CounterFlowOptions:
    width: float = 0.02
    grid: int = 20


def counterflow_flame(fuel: ct.Solution, oxidizer: ct.Solution, options: CounterFlowOptions = CounterFlowOptions()) -> ct.CounterflowDiffusionFlame:
    """Solve a simple counter-flow diffusion flame."""

    gas = ct.Solution('gri30.yaml') if isinstance(fuel, str) else fuel
    flame = ct.CounterflowDiffusionFlame(gas, width=options.width)
    flame.grid = np.linspace(0, options.width, options.grid)
    flame.set_initial_guess(fuel=fuel, oxidizer=oxidizer)
    flame.solve(loglevel=0, auto=True)
    return flame
