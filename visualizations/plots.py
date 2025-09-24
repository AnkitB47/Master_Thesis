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
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _downsample_markevery(n: int, target: int = 30) -> int:
    """Pick a sensible 'markevery' so ~target markers appear."""
    if n <= target:
        return 1
    step = int(np.ceil(n / target))
    return max(step, 2)


def _topn_species(cases: Sequence[dict], N: int = 6) -> list[str]:
    """Return top-N species ranked by peak |Y_full| across all cases."""

    if N <= 0:
        return []
    peaks: dict[str, float] = {}
    for case in cases:
        spec_map = case.get("species", {}) or {}
        for name, series in spec_map.items():
            if not series:
                continue
            full_vals = np.asarray(series[0], dtype=float)
            if full_vals.size == 0:
                continue
            peaks[name] = max(peaks.get(name, 0.0), float(np.nanmax(np.abs(full_vals))))
    ranked = sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:N]]


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
    fig, ax = plt.subplots(constrained_layout=True)
    for i, sp in enumerate(species):
        ax.plot(time_full, y_full[:, i], label=f"{sp} full")
        ax.plot(time_red,  y_red[:,  i], "--", label=f"{sp} red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mole fraction")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    _save(fig, out_base)


def plot_ignition_delays(delays: Sequence[float], labels: Sequence[str], out_base: str) -> None:
    """Bar chart of ignition delays (e.g., ['Full','Reduced'])."""
    fig, ax = plt.subplots(constrained_layout=True)
    bars = ax.bar(labels, delays)
    ax.set_ylabel("Ignition delay [s]")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, delays):
        ax.annotate(f"{v:.2e}", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    _save(fig, out_base)


def plot_convergence(histories: Sequence[Iterable[float]], labels: Sequence[str], out_base: str) -> None:
    """Plot convergence histories for metaheuristics."""
    fig, ax = plt.subplots(constrained_layout=True)
    for hist, label in zip(histories, labels):
        if hist:
            ax.plot(list(range(len(hist))), list(hist), label=label, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
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
        fig, ax = plt.subplots(constrained_layout=True)
        ax.text(0.5, 0.5, "No common species to plot", ha="center")
        _save(fig, out_base)
        return

    idxF = [names_full.index(s) for s in common]
    idxR = [names_red.index(s) for s in common]

    fig, ax = plt.subplots(constrained_layout=True)

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
    ax.legend(
        ncol=2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )
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
        fig, ax = plt.subplots(constrained_layout=True)
        ax.text(0.5, 0.5, "No common species", ha="center")
        _save(fig, out_base)
        return

    # interpolate reduced to full grid so ΔY aligns
    Y_red_interp = np.empty((len(time_full), len(common)))
    for j, s in enumerate(common):
        jF = names_full.index(s)
        jR = names_red.index(s)
        Y_red_interp[:, j] = np.interp(time_full, time_red, Y_red[:, jR])

    fig, ax = plt.subplots(constrained_layout=True)

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

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )

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

    fig, ax = plt.subplots(constrained_layout=True)
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
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
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
    fig, ax = plt.subplots(constrained_layout=True)
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
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )
    _save(fig, out_base)


def plot_axial_overlays(
    cases: Sequence[dict],
    species: Sequence[str] | None,
    out_base: str,
    *,
    topn: int = 6,
    chunk_size: int = 4,
) -> None:
    if not cases:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.text(0.5, 0.5, "No cases to plot", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_base)
        return

    species_list = list(species) if species else _topn_species(cases, N=topn)
    if not species_list:
        species_list = _topn_species(cases, N=topn)

    chunks = [cases[i : i + chunk_size] for i in range(0, len(cases), chunk_size)]
    for page, subset in enumerate(chunks, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        for data in subset:
            x = np.asarray(data.get("x", []), dtype=float)
            if x.size == 0:
                continue
            xu, idx = np.unique(x, return_index=True)
            T_full = np.asarray(data.get("T_full", []), dtype=float)
            T_red = np.asarray(data.get("T_red", []), dtype=float)
            valid = (idx < T_full.size) & (idx < T_red.size)
            if not np.any(valid):
                continue
            idx_valid = idx[valid]
            xu_valid = xu[valid]
            axes[0].plot(xu_valid, T_full[idx_valid], label=f"{data['id']} full")
            axes[0].plot(xu_valid, T_red[idx_valid], linestyle="--", label=f"{data['id']} red")
            axes[0].axvline(data.get("ign_full", np.nan), color="0.6", ls=":", alpha=0.5)
            axes[0].axvline(data.get("ign_red", np.nan), color="0.3", ls="--", alpha=0.5)

        axes[0].set_xlabel("Axial coordinate [m]")
        axes[0].set_ylabel("Temperature [K]")
        axes[0].grid(True, alpha=0.3)
        if subset:
            axes[0].legend(
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=8,
            )

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, sp in enumerate(species_list):
            color = colors[i % len(colors)]
            for data in subset:
                spec = (data.get("species") or {}).get(sp)
                if not spec:
                    continue
                x = np.asarray(data.get("x", []), dtype=float)
                if x.size == 0:
                    continue
                xu, idx = np.unique(x, return_index=True)
                full_arr = np.asarray(spec[0], dtype=float)
                red_arr = np.asarray(spec[1], dtype=float)
                valid = (idx < full_arr.size) & (idx < red_arr.size)
                if not np.any(valid):
                    continue
                idx_valid = idx[valid]
                xu_valid = xu[valid]
                axes[1].plot(
                    xu_valid,
                    full_arr[idx_valid],
                    color=color,
                    alpha=0.8,
                    label=f"{sp} {data['id']} full",
                )
                axes[1].plot(
                    xu_valid,
                    red_arr[idx_valid],
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                    label=f"{sp} {data['id']} red",
                )

        axes[1].set_xlabel("Axial coordinate [m]")
        axes[1].set_ylabel("Mass fraction")
        axes[1].grid(True, alpha=0.3)
        if subset:
            axes[1].legend(
                ncol=1,
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=7,
            )

        suffix = "" if len(chunks) == 1 else f"_page{page}"
        _save(fig, out_base + suffix)


def plot_kpi_bars(df, out_base: str) -> None:
    if df is None or df.empty:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.text(0.5, 0.5, "No KPI data", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_base)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    idx = np.arange(len(df))
    width = 0.35

    axes[0].bar(idx - width / 2, df["CH4_full"], width, label="Full")
    axes[0].bar(idx + width / 2, df["CH4_red"], width, label="Reduced")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(df["case_id"], rotation=45, ha="right")
    axes[0].set_ylabel("CH4 conversion")
    axes[0].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(idx - width / 2, df["H2CO_full"], width, label="Full")
    axes[1].bar(idx + width / 2, df["H2CO_red"], width, label="Reduced")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(df["case_id"], rotation=45, ha="right")
    axes[1].set_ylabel("H$_2$/CO ratio")
    axes[1].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )
    axes[1].grid(axis="y", alpha=0.3)

    _save(fig, out_base)


def plot_consistency_stub(df, out_base: str, baseline) -> None:
    fig, ax = plt.subplots(constrained_layout=True)
    if baseline is None or df is None or df.empty:
        ax.text(0.5, 0.5, "No 0D baseline available", ha="center", va="center")
        ax.axis("off")
        _save(fig, out_base)
        return

    ax.scatter(df["CH4_full"], df["CH4_red"], label="Cases")
    ax.plot([0, 1], [0, 1], "k--", label="1:1")
    ax.set_xlabel("1D full CH4 conversion")
    ax.set_ylabel("1D reduced CH4 conversion")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out_base)
