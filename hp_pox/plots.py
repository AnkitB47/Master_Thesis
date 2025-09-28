"""Plotting helpers for HP-POX diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
)


def plot_profile_overlay(
    positions: np.ndarray,
    temperature_full: np.ndarray,
    species_full: Mapping[str, np.ndarray],
    temperature_reduced: np.ndarray | None,
    species_reduced: Mapping[str, np.ndarray] | None,
    species: Sequence[str],
    out_path: Path,
    ignition_full: float | None = None,
    ignition_reduced: float | None = None,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    ax[0].plot(positions, temperature_full, label="full", linewidth=2)
    if temperature_reduced is not None:
        ax[0].plot(positions, temperature_reduced, label="reduced", linestyle="--")
    ax[0].set_xlabel("Axial position (m)")
    ax[0].set_ylabel("Temperature (K)")
    ax[0].legend()
    if ignition_full is not None:
        ax[0].axvline(
            ignition_full, color="tab:red", linestyle=":", label="ignition full"
        )
    if ignition_reduced is not None:
        ax[0].axvline(
            ignition_reduced,
            color="tab:orange",
            linestyle="--",
            label="ignition reduced",
        )
    ax[0].set_title("Axial temperature")

    for specie in species:
        profile_full = species_full.get(specie)
        if profile_full is None:
            continue
        ax[1].plot(positions, profile_full, label=f"{specie} (full)", linewidth=2)
        if species_reduced and specie in species_reduced:
            ax[1].plot(
                positions,
                species_reduced[specie],
                label=f"{specie} (reduced)",
                linestyle="--",
            )
    ax[1].set_xlabel("Axial position (m)")
    ax[1].set_ylabel("Mole fraction")
    ax[1].legend(ncol=2, fontsize=8)
    ax[1].set_title("Major species profiles")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_progress_variable_overlay(
    positions_full: np.ndarray,
    pv_full: np.ndarray,
    positions_reduced: np.ndarray,
    pv_reduced: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(positions_full, pv_full, label="full", linewidth=2)
    ax.plot(
        positions_reduced,
        pv_reduced,
        linestyle="--",
        marker="o",
        markersize=3,
        label="reduced",
    )
    ax.set_xlabel("Axial position (m)")
    ax.set_ylabel("Progress variable")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_profile_residuals(
    positions: np.ndarray,
    residuals: Mapping[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for specie, data in residuals.items():
        ax.plot(positions, data, label=specie)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Axial position (m)")
    ax.set_ylabel("ΔX")
    ax.set_title("Species residuals")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_timescales_overlay(
    time_full: np.ndarray,
    scales_full: Mapping[str, np.ndarray],
    time_reduced: np.ndarray,
    scales_reduced: Mapping[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for name, series in scales_full.items():
        ax.semilogy(time_full, series, label=f"{name} (full)")
    for name, series in scales_reduced.items():
        ax.semilogy(time_reduced, series, linestyle="--", label=f"{name} (reduced)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Timescale (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ignition_positions(
    full_position: float | None,
    reduced_position: float | None,
    out_path: Path,
    labels: tuple[str, str] = ("full", "reduced"),
) -> None:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    xpos = [0, 1]
    ypos = [
        full_position if full_position is not None else np.nan,
        reduced_position if reduced_position is not None else np.nan,
    ]
    ax.scatter(xpos, ypos, s=80)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Ignition position (m)")
    ax.set_ylim(bottom=0)
    ax.set_title("Ignition position comparison")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_kpi_summary(
    sweep_positions: Sequence[float],
    values: Mapping[str, Sequence[float]],
    out_path: Path,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for name, series in values.items():
        ax.plot(sweep_positions, series, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.grid(True, linewidth=0.6, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_placeholder(message: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
