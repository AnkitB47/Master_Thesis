"""Computation of PVTS and SPTS time scales for batch reactors."""

from __future__ import annotations

from typing import Sequence, Tuple, List

import numpy as np


def pv_timescale(
    time: np.ndarray,
    Y: np.ndarray,
    names: Sequence[str],
    pv_species: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return progress variable and its characteristic time scale.

    Parameters
    ----------
    time:
        1D array of time samples.
    Y:
        Mass-fraction array of shape ``(len(time), nspecies)`` aligned with
        ``names``.
    names:
        Species names corresponding to columns of ``Y``.
    pv_species:
        Species to include in the progress variable. Defaults to a common set
        of products and radicals used for methane combustion.

    Returns
    -------
    pv : np.ndarray
        Progress variable PV(t).
    tau : np.ndarray
        PVTS characteristic time scale :math:`|ΔPV|/|dPV/dt|` evaluated at each
        time point.
    """

    time = np.asarray(time, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if pv_species is None:
        pv_species = ["CO2", "H2O", "CO", "H2", "O", "H", "OH"]

    # enforce strictly increasing time to stabilise gradients
    eps = 1e-12
    time_mono = time.copy()
    for i in range(1, len(time_mono)):
        if time_mono[i] <= time_mono[i - 1]:
            time_mono[i] = time_mono[i - 1] + eps

    idx = [names.index(s) for s in pv_species if s in names]
    if not idx:
        pv = np.zeros_like(time_mono)
    else:
        pv = Y[:, idx].sum(axis=1)

    if time_mono.size < 3:
        dPVdt = np.gradient(pv, time_mono, edge_order=1)
    else:
        dPVdt = np.gradient(pv, time_mono, edge_order=2)
    delta = pv[-1] - pv
    tau = np.abs(delta) / (np.abs(dPVdt) + 1e-30)
    return pv, tau


def spts(time: np.ndarray, Y: np.ndarray) -> np.ndarray:
    r"""Return scalar progress time scale (SPTS).

    The SPTS is approximated from finite differences as

    .. math::

        \tau_{SPTS} = \frac{\|Y^* - Y(t)\|_2}{\|dY/dt\|_2}

    where ``Y*`` is the final state.
    """

    time = np.asarray(time, dtype=float)
    Y = np.asarray(Y, dtype=float)
    eps = 1e-12
    time_mono = time.copy()
    for i in range(1, len(time_mono)):
        if time_mono[i] <= time_mono[i - 1]:
            time_mono[i] = time_mono[i - 1] + eps

    if time_mono.size < 3:
        dYdt = np.gradient(Y, time_mono, axis=0, edge_order=1)
    else:
        dYdt = np.gradient(Y, time_mono, axis=0, edge_order=2)
    delta = Y[-1] - Y
    norm_delta = np.linalg.norm(delta, axis=1)
    norm_dYdt = np.linalg.norm(dYdt, axis=1)
    tau = np.abs(norm_delta) / (np.abs(norm_dYdt) + 1e-30)
    return tau


__all__ = ["pv_timescale", "spts"]

