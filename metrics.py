"""Distance metrics for reactor comparison."""

from __future__ import annotations

import numpy as np


def l2_distance(Y_full: np.ndarray, Y_red: np.ndarray, time: np.ndarray) -> float:
    """Integrate squared difference over time."""
    diff = Y_full - Y_red
    return float(np.trapz(np.sum(diff**2, axis=1), time))
