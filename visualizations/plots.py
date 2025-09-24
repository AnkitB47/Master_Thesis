from __future__ import annotations
"""Plotting utilities for mechanism reduction results (paper-ready)."""

import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# small helpers
# -------------------------
def _save(fig: plt.Figure, out_base: str) -> None:
    """Save figure as PNG and PDF (pub-ready)."""
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _downsample_markevery(n: int, target: int = 30) -> int:
    """Pick a sensible 'markevery' so ~target markers appear."""
    if n <= target:
        return 1
    step = int(np.ceil(n / target))
    return max(step, 2)


# -------------------------
# legacy/simple plots
# -------------------------
def plot_mole_fraction(
    time_full: np.ndarray,
    y_full: np.ndarray,
    time_red: np.ndarray,
    y_red: np.ndarray,
    species: Sequence[str],
    out_base: str,
) -> None:
    """(Kept for backward-compat; simple overlay)."""
    fig, ax = plt.subplots()
    for i, sp in enumerate(species):
        ax.plot(time_full, y_full[:, i], label=f"{sp} full")
        ax.plot(time_red,  y_red[:,  i], "--", label=f"{sp} red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mole fraction")
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_base)


def plot_ignition_delays(delays: Sequence[float], labels: Sequence[str], out_base: str) -> None:
    """Bar chart of ignition delays (e.g., ['Full','Reduced'])."""
    fig, ax = plt.subplots()
    bars = ax.bar(labels, delays)
    ax.set_ylabel("Ignition delay [s]")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, delays):
        ax.annotate(f"{v:.2e}", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    _save(fig, out_base)


def plot_convergence(histories: Sequence[Iterable[float]], labels: Sequence[str], out_base: str) -> None:
    """Plot convergence histories for metaheuristics."""
    fig, ax = plt.subplots()
    for hist, label in zip(histories, labels):
        if hist:
            ax.plot(list(range(len(hist))), list(hist), label=label, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, out_base)


# -------------------------
# paper-style profiles
# -------------------------
# (fixed colors per species so legend पढ़ते ही पहचान बने)
_COLOR = {
    "O2":  "#1f77b4",  # blue
    "CO2": "#2ca02c",  # green
    "H2O": "#d62728",  # red
    "CO":  "#17becf",  # teal
    "CH4": "#9467bd",  # purple
    "OH":  "#8c564b",  # brown
}

def _style_color(name: str, fallback: str) -> str:
    return _COLOR.get(name, fallback)


def plot_species_profiles(
    time_full: np.ndarray,
    Y_full: np.ndarray,
    names_full: Sequence[str],
    time_red: np.ndarray,
    Y_red: np.ndarray,
    names_red: Sequence[str],
    species: Sequence[str],
    out_base: str,
    *,
    tau_full: float | None = None,
    tau_red: float | None = None,
    focus: str | None = None,
    focus_window: tuple[float, float] | None = None,
) -> None:
    """
    Paper-style overlay of species profiles.
    - full = solid line
    - reduced = hollow circle markers (same color)
    - log x-axis; optional zoom around ignition delay `tau_full`
    - optional y-limits for a fixed vertical range (e.g., (-0.02, 0.45))
    """
    # common species in the requested order
    common = [s for s in species if s in names_full and s in names_red]
    if not common:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No common species to plot", ha="center")
        _save(fig, out_base)
        return

    idxF = [names_full.index(s) for s in common]
    idxR = [names_red.index(s) for s in common]

    fig, ax = plt.subplots()

    if tau_full is not None and np.isfinite(tau_full) and tau_full > 0:
        ax.axvline(tau_full, color="0.2", ls="--", lw=1, zorder=0)
    if tau_red is not None and np.isfinite(tau_red) and tau_red > 0:
        ax.axvline(tau_red, color="0.6", ls="--", lw=1, zorder=0)

    tmin = tmax = None
    if focus == "auto" and tau_full is not None and np.isfinite(tau_full):
        tmin, tmax = 0.8 * tau_full, 1.6 * tau_full
    elif focus == "window" and focus_window is not None:
        tmin, tmax = focus_window

    if tmin is not None and tmax is not None:
        ax.set_xlim(tmin, tmax)
        mask = (time_full >= tmin) & (time_full <= tmax)
        Y_red_int = np.empty((mask.sum(), len(common)))
        for j, s in enumerate(common):
            Y_red_int[:, j] = np.interp(time_full[mask], time_red, Y_red[:, idxR[j]])
        yvals = np.concatenate([Y_full[mask][:, idxF], Y_red_int], axis=1)
        ymin, ymax = float(yvals.min()), float(yvals.max())
        pad = 0.1 * (ymax - ymin if ymax > ymin else 1e-6)
        ax.set_ylim(ymin - pad, ymax + pad)
        mk_every = 2
    else:
        mk_every = _downsample_markevery(len(time_red))

    for i, s in enumerate(common):
        color = _style_color(s, fallback=f"C{i}")
        ax.semilogx(time_full, Y_full[:, idxF[i]], linewidth=2.2, color=color, label=f"{s} full")
        ax.semilogx(
            time_red,
            Y_red[:, idxR[i]],
            linestyle="none",
            marker="o",
            markersize=4.0,
            fillstyle="none",
            color=color,
            markevery=mk_every,
            label=f"{s} reduced",
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass fraction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, frameon=False)
    _save(fig, out_base)


def plot_species_residuals(
    time_full: np.ndarray,
    Y_full: np.ndarray,
    time_red: np.ndarray,
    Y_red: np.ndarray,
    names_full: Sequence[str],
    names_red: Sequence[str],
    species: Sequence[str],
    out_base: str,
    *,
    tau_full: float | None = None,
    focus: str | None = None,
    focus_window: tuple[float, float] | None = None,
) -> None:
    """Plot residuals ``Y_full(t) - Y_red(t)`` on full time grid (reduced is interpolated)."""
    common = [s for s in species if s in names_full and s in names_red]
    if not common:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No common species", ha="center")
        _save(fig, out_base)
        return

    # interpolate reduced to full grid so ΔY aligns
    Y_red_interp = np.empty((len(time_full), len(common)))
    for j, s in enumerate(common):
        jF = names_full.index(s)
        jR = names_red.index(s)
        Y_red_interp[:, j] = np.interp(time_full, time_red, Y_red[:, jR])

    fig, ax = plt.subplots()

    tmin = tmax = None
    if focus == "auto" and tau_full is not None and np.isfinite(tau_full):
        tmin, tmax = 0.8 * tau_full, 1.6 * tau_full
    elif focus == "window" and focus_window is not None:
        tmin, tmax = focus_window

    mask = (time_full >= tmin) & (time_full <= tmax) if tmin is not None else slice(None)
    tplot = time_full[mask]
    YF = Y_full[mask]
    YR = Y_red_interp[mask]

    for j, s in enumerate(common):
        color = _style_color(s, fallback=f"C{j}")
        ax.semilogx(tplot, YF[:, names_full.index(s)] - YR[:, j], label=s, color=color)

    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ΔY (full - red)")
    ax.grid(True, which="both", alpha=0.3)
    if tmin is not None:
        ax.set_xlim(tmin, tmax)

    leg = ax.legend(frameon=False, ncol=2)
    fig.canvas.draw()
    if leg.get_window_extent().overlaps(ax.get_window_extent()):
        leg.remove()
        ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")

    _save(fig, out_base)


def plot_progress_variable(
    time_full: np.ndarray,
    pv_full: np.ndarray,
    time_red: np.ndarray,
    pv_red: np.ndarray,
    out_base: str,
    *,
    tau: float | None = None,
    tau_full: float | None = None,
    focus: str | None = None,
    focus_window: tuple[float, float] | None = None,
) -> None:
    """Overlay progress variables for full vs. reduced mechanisms."""
    if tau_full is None:
        tau_full = tau

    fig, ax = plt.subplots()
    ax.semilogx(time_full, pv_full, label="PV full", linewidth=2)
    mk_every = _downsample_markevery(len(time_red))
    ax.semilogx(
        time_red,
        pv_red,
        linestyle="none",
        marker="o",
        fillstyle="none",
        markevery=mk_every,
        label="PV reduced",
    )
    if tau_full is not None and np.isfinite(tau_full) and tau_full > 0:
        ax.axvline(tau_full, color="0.4", ls="--", lw=1)

    tmin = tmax = None
    if focus == "auto" and tau_full is not None and np.isfinite(tau_full):
        tmin, tmax = 0.8 * tau_full, 1.6 * tau_full
    elif focus == "window" and focus_window is not None:
        tmin, tmax = focus_window
    if tmin is not None:
        ax.set_xlim(tmin, tmax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Progress variable")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, out_base)


def plot_timescales(
    time_full: np.ndarray,
    tau_pv_full: np.ndarray,
    tau_spts_full: np.ndarray,
    time_red: np.ndarray,
    tau_pv_red: np.ndarray,
    tau_spts_red: np.ndarray,
    out_base: str,
) -> None:
    """Overlay PVTS and SPTS time scales for full and reduced cases."""
    fig, ax = plt.subplots()
    mk_every = _downsample_markevery(len(time_red))
    ax.semilogx(time_full, tau_pv_full, label="PVTS full", linewidth=2)
    ax.semilogx(time_red, tau_pv_red, linestyle="--", marker="o",
                markevery=mk_every, fillstyle="none", label="PVTS reduced")
    ax.semilogx(time_full, tau_spts_full, label="SPTS full", linewidth=2)
    ax.semilogx(time_red, tau_spts_red, linestyle="--", marker="s",
                markevery=mk_every, fillstyle="none", label="SPTS reduced")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Time scale [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_base)


def plot_axial_overlays(cases: Sequence[dict], species: Sequence[str], out_base: str) -> None:
    if not cases:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cases to plot", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_base)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for data in cases:
        x = np.asarray(data.get("x", []), dtype=float)
        if x.size == 0:
            continue
        axes[0].plot(x, data.get("T_full", []), label=f"{data['id']} full")
        axes[0].plot(x, data.get("T_red", []), linestyle="--", label=f"{data['id']} red")
        axes[0].axvline(data.get("ign_full", np.nan), color="0.6", ls=":", alpha=0.5)
        axes[0].axvline(data.get("ign_red", np.nan), color="0.3", ls="--", alpha=0.5)

    axes[0].set_xlabel("Axial coordinate [m]")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].grid(True, alpha=0.3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, sp in enumerate(species):
        color = colors[i % len(colors)]
        for data in cases:
            spec = data.get("species", {}).get(sp)
            if not spec:
                continue
            axes[1].plot(data["x"], spec[0], color=color, alpha=0.8, label=f"{sp} {data['id']} full")
            axes[1].plot(data["x"], spec[1], color=color, linestyle="--", alpha=0.6, label=f"{sp} {data['id']} red")

    axes[1].set_xlabel("Axial coordinate [m]")
    axes[1].set_ylabel("Mass fraction")
    if cases:
        axes[1].legend(ncol=2, frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    _save(fig, out_base)


def plot_kpi_bars(df, out_base: str) -> None:
    if df is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No KPI data", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_base)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    idx = np.arange(len(df))
    width = 0.35

    axes[0].bar(idx - width / 2, df["CH4_full"], width, label="Full")
    axes[0].bar(idx + width / 2, df["CH4_red"], width, label="Reduced")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(df["case_id"], rotation=45, ha="right")
    axes[0].set_ylabel("CH4 conversion")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(idx - width / 2, df["H2CO_full"], width, label="Full")
    axes[1].bar(idx + width / 2, df["H2CO_red"], width, label="Reduced")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(df["case_id"], rotation=45, ha="right")
    axes[1].set_ylabel("H$_2$/CO ratio")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_base)


def plot_consistency_stub(df, out_base: str, baseline) -> None:
    fig, ax = plt.subplots()
    if baseline is None or df is None or df.empty:
        ax.text(0.5, 0.5, "No 0D baseline available", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_base)
        return

    ax.scatter(df["CH4_full"], df["CH4_red"], label="Cases")
    ax.plot([0, 1], [0, 1], "k--", label="1:1")
    ax.set_xlabel("1D full CH4 conversion")
    ax.set_ylabel("1D reduced CH4 conversion")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, out_base)
