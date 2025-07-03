"""Utility functions for progress variable calculations."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence


def progress_variable(Y: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Return :math:`PV(t)=\sum_i w_i Y_i(t)`.

    Parameters
    ----------
    Y:
        Matrix of mass fractions with shape ``(n_steps, n_species)``.
    weights:
        Sequence of species weights ``w_i``.

    Returns
    -------
    np.ndarray
        Progress variable as a one-dimensional array of length ``n_steps``.
    """

    W = np.asarray(weights, dtype=float)
    if Y.shape[1] != W.size:
        raise ValueError("Weight length must match number of species")
    return (Y * W[None, :]).sum(axis=1)


def optimise_weights(
    Y: np.ndarray,
    pv_ref: np.ndarray,
    eval_options: "GAOptions | None" = None,
) -> np.ndarray:
    """Optimise progress-variable weights via a genetic algorithm.

    The optimisation minimises the L2 error between ``progress_variable(Y, w)``
    and a reference progress variable ``pv_ref``.

    Parameters
    ----------
    Y:
        Mass-fraction matrix ``(n_steps, n_species)``.
    pv_ref:
        Reference progress variable of length ``n_steps``.
    eval_options:
        Optional :class:`~metaheuristics.ga.GAOptions` to control the GA.

    Returns
    -------
    np.ndarray
        Optimised weight vector ``w``.
    """

    from metaheuristics.ga import GAOptions, run_ga

    options = eval_options or GAOptions()

    genome_length = Y.shape[1]

    def eval_fn(genome: np.ndarray) -> float:
        # map binary genome to weights in [0, 1]
        w = genome.astype(float)
        pv = progress_variable(Y, w)
        err = np.linalg.norm(pv - pv_ref) / (np.linalg.norm(pv_ref) + 1e-12)
        # GA maximises fitness, so return negative error
        return -err

    best, _ = run_ga(genome_length, eval_fn, options)
    return best.astype(float)

