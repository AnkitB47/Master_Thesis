"""Plotting utilities for metaheuristic convergence and comparison charts."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_convergence(histories: Sequence[Sequence[float]], names: Sequence[str], out_dir: str) -> None:
    """Plot fitness convergence for each algorithm.

    Parameters
    ----------
    histories:
        Sequence of fitness histories (one list per algorithm).
    names:
        Algorithm names in the same order as ``histories``.
    out_dir:
        Directory to save plots. PNG and PDF are saved with the name
        ``convergence``.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for hist, name in zip(histories, names):
        if len(hist) == 0:
            continue
        plt.plot(range(len(hist)), hist, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    png = os.path.join(out_dir, "convergence.png")
    pdf = os.path.join(out_dir, "convergence.pdf")
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()


def plot_comparison(metrics_csv: str, out_dir: str) -> None:
    """Create comparison charts for reduction performance.

    Parameters
    ----------
    metrics_csv:
        Path to CSV file with columns ``algorithm``, ``pv_error``, ``ignition_delay_error``,
        ``size_reduction`` and ``runtime``.
    out_dir:
        Directory to save plots. PNG and PDF are saved with the base name
        ``comparison``.
    """
    df = pd.read_csv(metrics_csv)
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))

    axes[0].bar(df["algorithm"], df["size_reduction"])
    axes[0].set_ylabel("Size Reduction")

    axes[1].bar(df["algorithm"], df["runtime"])
    axes[1].set_ylabel("Runtime [s]")

    axes[2].bar(df["algorithm"], df["pv_error"])
    axes[2].set_ylabel("PV Error")
    axes[2].set_xlabel("Algorithm")

    fig.tight_layout()
    png = os.path.join(out_dir, "comparison.png")
    pdf = os.path.join(out_dir, "comparison.pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
