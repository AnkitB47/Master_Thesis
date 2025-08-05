"""Distance metrics for reactor comparison."""

from __future__ import annotations

import numpy as np


def l2_distance(Y_full: np.ndarray, Y_red: np.ndarray, time: np.ndarray) -> float:
    """Integrate squared difference over time."""
    diff = Y_full - Y_red
    return float(np.trapz(np.sum(diff**2, axis=1), time))


def pv_error(full: np.ndarray, red: np.ndarray) -> float:
    """Relative L2 error between two progress variables."""
    return float(np.linalg.norm(full - red) / (np.linalg.norm(full) + 1e-12))


def ignition_delay(
    time: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray | None = None,
    species_idx: int | None = None,
) -> float:
    """Estimate ignition delay from temperature or species profiles."""

    if species_idx is not None and Y is not None:
        deriv = np.gradient(Y[:, species_idx], time)
    else:
        deriv = np.gradient(T, time)
    idx = int(np.argmax(deriv))
    return float(time[idx])
