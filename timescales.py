"""Timescale computations from the system Jacobian."""

from __future__ import annotations

import cantera as ct
import numpy as np


def species_timescales(gas: ct.Solution) -> np.ndarray:
    """Return characteristic time scales from the Jacobian eigenvalues."""
    J = gas.jacobian_full
    eig = np.linalg.eigvals(J)
    with np.errstate(divide="ignore"):
        tau = 1 / np.abs(eig)
    return np.sort(np.real(tau))


def progress_variable_timescale(pv: np.ndarray, time: np.ndarray) -> float:
    """Estimate PV timescale as the time for PV to rise from 10% to 90%."""
    pv_norm = (pv - pv.min()) / (pv.max() - pv.min() + 1e-12)
    i1 = np.searchsorted(pv_norm, 0.1)
    i2 = np.searchsorted(pv_norm, 0.9)
    if i2 <= i1:
        return 0.0
    return float(time[i2] - time[i1])
