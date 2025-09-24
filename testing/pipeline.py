import os
import json
import csv
import logging
import numpy as np
import pandas as pd
import cantera as ct

from typing import Sequence, Dict, Tuple, List, Callable
from reactor import PFRRunner, PFRConfig

from mechanism.loader import Mechanism
from mechanism.mix import methane_air_mole_fractions, methane_oxygen_mole_fractions, mole_to_mass_fractions, HR_PRESETS
from reactor.batch import BatchResult, run_constant_pressure, run_isothermal_const_p
from metaheuristics.ga import run_ga, GAOptions
from progress_variable import PV_SPECIES_DEFAULT, pv_error_aligned
from metrics import ignition_delay
from graph.construction import build_species_graph
from gnn.models import train_gnn, predict_scores
from visualizations import (
    plot_ignition_delays,
    plot_convergence,
    plot_species_profiles,
    plot_species_residuals,
    plot_progress_variable,
    plot_timescales,
    plot_axial_overlays,
    plot_kpi_bars,
    plot_consistency_stub,
)
from timescales import pv_timescale, spts


# --- Alignment utilities ---
import numpy as _np


def _align_to_reference(t_ref: _np.ndarray,
                        t_a: _np.ndarray, a: _np.ndarray,
                        t_b: _np.ndarray, b: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Resample series 'a' and 'b' onto the reference time grid t_ref using 1D linear interpolation.
    Handles scalar, 1D, or matching-length 1D arrays.
    Returns (a_ref, b_ref) with shape (len(t_ref),) each.
    """
    t_ref = _np.asarray(t_ref, dtype=float)
    t_a = _np.asarray(t_a, dtype=float)
    t_b = _np.asarray(t_b, dtype=float)
    a = _np.asarray(a, dtype=float)
    b = _np.asarray(b, dtype=float)

    # Edge cases: if a/b are scalar, broadcast
    if a.ndim == 0:
        a_ref = _np.full_like(t_ref, float(a))
    else:
        a_ref = _np.interp(t_ref, t_a, a, left=a[0], right=a[-1])

    if b.ndim == 0:
        b_ref = _np.full_like(t_ref, float(b))
    else:
        b_ref = _np.interp(t_ref, t_b, b, left=b[0], right=b[-1])

    return a_ref, b_ref


def _ensure_same_grid(time_ref: _np.ndarray,
                      series_pairs: list[tuple[_np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray]]) -> list[tuple[_np.ndarray, _np.ndarray]]:
    """
    For a list of pairs (t1, s1, t2, s2), resample both s1 and s2 to 'time_ref' with _align_to_reference.
    Returns list of (s1_ref, s2_ref).
    """
    out = []
    for (t1, s1, t2, s2) in series_pairs:
        a_ref, b_ref = _align_to_reference(time_ref, t1, s1, t2, s2)
        out.append((a_ref, b_ref))
    return out


def _interp_to(ref_t, y_t, y_vals):
    """Interpolate ``y_vals`` defined on ``y_t`` onto ``ref_t`` column-wise."""
    ref_t = np.asarray(ref_t, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if y_vals.ndim == 1:
        return np.interp(ref_t, y_t, y_vals)
    out = np.zeros((len(ref_t), y_vals.shape[1]), dtype=float)
    for j in range(y_vals.shape[1]):
        out[:, j] = np.interp(ref_t, y_t, y_vals[:, j])
    return out


def _rms_norm(arr):
    arr = np.asarray(arr, dtype=float)
    return np.linalg.norm(arr) / max(1, np.sqrt(arr.size))


logger = logging.getLogger(__name__)


# -------------------------------
# Helpers: label building for GNN
# -------------------------------

def _pv_weights_for(mech: Mechanism) -> np.ndarray:
    """Build PV weights aligned to mech.species_names."""
    W = np.zeros(len(mech.species_names), dtype=float)
    path = "data/species_weights.json"
    if os.path.exists(path):
        with open(path) as f:
            wmap = json.load(f)
        for i, s in enumerate(mech.species_names):
            W[i] = float(wmap.get(s, 0.0))
    else:
        # If file missing, put unit weights for a sensible default PV set
        for i, s in enumerate(mech.species_names):
            if s in {"CO2", "H2O", "CO", "H2", "O", "H", "OH"}:
                W[i] = 1.0
    # Avoid all-zeros
    if not np.any(W):
        W[:] = 1.0
    return W


def _baseline_short_run(
    mech: Mechanism,
    T0: float,
    p0: float,
    Y0: Dict[str, float],
    tf_short: float,
    steps_short: int,
    log_times: bool,
    runner,
) -> BatchResult:
    """Run a short baseline simulation with retries if inert.

    Some mixtures (especially at lean or rich limits) may not react within the
    short window used for LOO scoring which originally caused the GA to crash
    with ``RuntimeError('no_reaction')``.  To make the baseline more robust we
    gradually relax the integration settings:

    1. first attempt uses the provided ``tf_short``/``steps_short`` and ``T0``;
    2. on failure, retry with a 4× longer window and 2× more steps;
    3. if still inert, increase the initial temperature by 100 K.

    This mirrors the behaviour described in the user instructions and ensures
    that GA/LOO evaluations always have a baseline to compare against.
    """

    def _attempt(tf_s, steps_s, T_adj):
        return runner(
            mech.solution,
            T_adj,
            p0,
            Y0,
            tf_s,
            nsteps=steps_s,
            use_mole=False,
            log_times=log_times,
        )

    try:
        return _attempt(tf_short, steps_short, T0)
    except RuntimeError as e:
        if "no_reaction" not in str(e):
            raise
        try:
            # Longer window and more steps
            return _attempt(4 * tf_short, max(steps_short * 2, steps_short + 50), T0)
        except RuntimeError as e2:
            if "no_reaction" not in str(e2):
                raise
            # Last resort: bump the temperature slightly
            return _attempt(
                4 * tf_short,
                max(steps_short * 2, steps_short + 50),
                T0 + 100.0,
            )


def _loo_scores(
    base_mech: Mechanism,
    baseline: BatchResult,
    T0: float,
    p0: float,
    Y0: Dict[str, float],
    tf_short: float,
    steps_short: int,
    log_times: bool,
    critical: Sequence[str],
    runner,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Leave-one-out scores and post-ignition residual integrals for each species.
    Returns two dictionaries keyed by species name.
    """
    logger.info("Building LOO scores on short window: tf=%.3e, steps=%d", tf_short, steps_short)
    W = _pv_weights_for(base_mech)

    tau_short = baseline.ignition_delay or ignition_delay(baseline.time, baseline.temperature)[0]
    mask_post = baseline.time > tau_short

    scores: Dict[str, float] = {}
    resid: Dict[str, float] = {}
    for s in base_mech.species_names:
        if s in critical:
            scores[s] = 1.0
            resid[s] = 0.0
            continue
        try:
            m = Mechanism(base_mech.file_path)
            m.remove_species([s])
            red = _baseline_short_run(m, T0, p0, Y0, tf_short, steps_short, log_times, runner)

            red_interp = _interp_to(baseline.time, red.time, red.mass_fractions)
            score = pv_error_aligned(
                baseline.mass_fractions,
                red_interp,
                base_mech.species_names,
                m.species_names,
                W,
            )

            Y_red_interp = np.zeros_like(baseline.mass_fractions)
            map_red = {name: idx for idx, name in enumerate(m.species_names)}
            for j, name in enumerate(base_mech.species_names):
                idx = map_red.get(name)
                if idx is not None:
                    Y_red_interp[:, j] = red_interp[:, idx]

            diff = np.abs(baseline.mass_fractions[mask_post] - Y_red_interp[mask_post])
            resid_val = float(
                np.trapz((diff * W[None, :]).sum(axis=1), baseline.time[mask_post])
            )

            if not np.isfinite(score):
                score = 0.0
            if not np.isfinite(resid_val):
                resid_val = 0.0
        except Exception as e:
            logger.debug("LOO failed for %s: %s", s, e)
            score = 0.0
            resid_val = 0.0
        scores[s] = float(score)
        resid[s] = resid_val

    # Normalize to [0,1]
    vals = np.array(list(scores.values()), dtype=float)
    vmax = float(vals.max()) if vals.size else 0.0
    if vmax > 0:
        for k in scores:
            scores[k] = scores[k] / vmax

    rv = np.array(list(resid.values()), dtype=float)
    rmax = float(rv.max()) if rv.size else 0.0
    if rmax > 0:
        for k in resid:
            resid[k] = resid[k] / rmax

    return scores, resid


def _centrality_scores(G, mech: Mechanism) -> Dict[str, float]:
    # Simple degree centrality (fast, stable)
    deg = {n: float(G.degree(n)) for n in G.nodes}
    vmax = max(deg.values()) if deg else 1.0
    return {n: (deg[n] / vmax if vmax > 0 else 0.0) for n in mech.species_names}


def _build_species_labels(
    mech: Mechanism,
    G,
    full_short: BatchResult,
    T0: float,
    p0: float,
    Y0: Dict[str, float],
    tf_short: float,
    steps_short: int,
    log_times: bool,
    alpha: float,
    critical: Sequence[str],
    cache_labels: bool,
    runner,
    cache_dir: str = "results",
) -> Dict[str, float]:
    """
    Blend LOO and centrality to produce labels in [0,1]. Cache to JSON.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "species_labels.json")
    if cache_labels and os.path.exists(cache_path):
        with open(cache_path) as f:
            labels = json.load(f)
        logger.info("Loaded cached labels from %s", cache_path)
        return {k: float(v) for k, v in labels.items()}

    # Compute fresh
    loo, resid = _loo_scores(
        mech, full_short, T0, p0, Y0, tf_short, steps_short, log_times, critical, runner
    )
    cen = _centrality_scores(G, mech)

    labels: Dict[str, float] = {}
    for s in mech.species_names:
        base = alpha * loo.get(s, 0.0) + (1.0 - alpha) * cen.get(s, 0.0)
        labels[s] = float(base * (1.0 + 0.5 * resid.get(s, 0.0)))

    # Renormalize to [0,1]
    vals = np.array(list(labels.values()), dtype=float)
    vmax = float(vals.max()) if vals.size else 0.0
    if vmax > 0:
        for k in labels:
            labels[k] = labels[k] / vmax

    # Ensure critical are high
    for s in critical:
        labels[s] = 1.0

    with open(cache_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info("Wrote labels to %s", cache_path)
    return labels


# -------------------------------
# GA evaluation
# -------------------------------

def evaluate_selection(
    selection,
    base_mech,
    Y0,
    tf,
    full_res,
    weights,
    critical_idxs,
    T0,
    p0,
    steps,
    log_times,
    runner,
    tau_full,
    tau_pv_full,
    tau_spts_full,
    target_species: int | None,
    fitness_mode: str = "standard",
    tol_pv: float = 0.05,
    tol_delay: float = 0.05,
    tol_timescale: float = 0.05,
    tol_resid: float = 0.05,
    mode: str = "0d",
    zeta: float = 20.0,
):
    reason = ""
    _DEBUG_ALIGN = False

    if critical_idxs and (selection.sum() < max(4, len(critical_idxs)) or np.any(selection[critical_idxs] == 0)):
        info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, "missing_critical", int(selection.sum()), 0.0, 0.0, 0.0)
        return -1e6, *info

    sel = selection.copy()
    attempt = 0
    while True:
        keep = [base_mech.species_names[i] for i, bit in enumerate(sel) if bit]
        mech = Mechanism(base_mech.file_path)
        remove = [s for s in mech.species_names if s not in keep]

        try:
            if remove:
                mech.remove_species(remove)
            res = runner(
                mech.solution,
                T0,
                p0,
                Y0,
                tf,
                nsteps=steps,
                use_mole=False,
                log_times=log_times,
                time_grid=full_res.time,
            )
            if _DEBUG_ALIGN:
                print(
                    "len(full.time)=", len(full_res.time),
                    "len(res.time)=", len(getattr(res, "time", [])),
                )
            if hasattr(res, "time") and len(res.time) != len(full_res.time):
                # Force align all reduced quantities to the full grid later
                pass  # alignment utilities will handle it; this is just a reminder checkpoint.
        except RuntimeError as e:
            if "no_reaction" in str(e) and attempt == 0:
                sel = sel.copy()
                sel[critical_idxs] = 1
                attempt += 1
                continue
            info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, str(e), int(sel.sum()), 0.0, 0.0, 0.0)
            return -1e6, *info
        except Exception as e:
            info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, f"sim_failed:{type(e).__name__}", int(sel.sum()), 0.0, 0.0, 0.0)
            return -1e6, *info
        break

    # interpolate reduced onto full time grid for penalties
    Y_red_interp = _interp_to(full_res.time, res.time, res.mass_fractions)
    weights_arr = np.asarray(weights, dtype=float)

    err = pv_error_aligned(
        full_res.mass_fractions,
        Y_red_interp,
        base_mech.species_names,
        mech.species_names,
        weights_arr,
    )
    delay_red, _ = ignition_delay(res.time, res.temperature)
    delay_diff = abs(delay_red - tau_full) / max(tau_full, 1e-12)
    x_full = getattr(full_res, "x", None)
    x_red = getattr(res, "x", None)
    ign_shift = 0.0
    delay_metric = delay_diff
    if mode == "1d" and x_full is not None and x_red is not None:
        x_full_arr = np.asarray(x_full, dtype=float)
        x_red_arr = np.asarray(x_red, dtype=float)
        try:
            x_ign_full = float(np.interp(tau_full, full_res.time, x_full_arr))
            x_ign_red = float(np.interp(delay_red, res.time, x_red_arr))
            ign_shift = abs(x_ign_red - x_ign_full) / max(x_ign_full, 1e-12)
            delay_metric = 0.5 * (delay_diff + ign_shift)
        except Exception:  # pragma: no cover (fallback when interpolation fails)
            ign_shift = 0.0
            delay_metric = delay_diff
    delta_T = float(res.temperature[-1] - res.temperature[0])
    delta_Y = float(np.max(np.abs(res.mass_fractions[-1] - res.mass_fractions[0])))

    # timescale mismatch
    _, tau_pv_red = pv_timescale(res.time, res.mass_fractions, mech.species_names)
    tau_spts_red = spts(res.time, res.mass_fractions)

    log_full_pv = np.log10(tau_pv_full + 1e-30)
    log_red_pv = _interp_to(full_res.time, res.time, np.log10(tau_pv_red + 1e-30))
    log_full_spts = np.log10(tau_spts_full + 1e-30)
    log_red_spts = _interp_to(full_res.time, res.time, np.log10(tau_spts_red + 1e-30))

    tau_mis = _rms_norm(log_full_pv - log_red_pv) + _rms_norm(log_full_spts - log_red_spts)

    # post-ignition species penalty
    map_full = {s: i for i, s in enumerate(base_mech.species_names)}
    map_red = {s: i for i, s in enumerate(mech.species_names)}
    Y_red_full = np.zeros_like(full_res.mass_fractions)
    for s, idx_full in map_full.items():
        idx_red = map_red.get(s)
        if idx_red is not None:
            Y_red_full[:, idx_full] = Y_red_interp[:, idx_red]
    mask_post = full_res.time > tau_full
    if mode == "1d" and x_full is not None:
        x_full_arr = np.asarray(x_full, dtype=float)
        if x_full_arr.size >= 2:
            x_mono = np.maximum.accumulate(x_full_arr)
        else:
            x_mono = x_full_arr
        try:
            x_ign_full = float(np.interp(tau_full, full_res.time, x_mono))
        except Exception:
            x_ign_full = x_mono[x_mono.size // 10] if x_mono.size else 0.0
        if x_mono.size:
            L = float(x_mono[-1])
        else:
            L = 1.0
        upper = min(L, x_ign_full + 0.3 * L)
        alt = (x_mono >= x_ign_full) & (x_mono <= upper)
        if np.any(alt):
            mask_post = alt
    if isinstance(mask_post, np.ndarray) and not np.any(mask_post):
        mask_post = full_res.time > tau_full
        if isinstance(mask_post, np.ndarray) and not np.any(mask_post):
            mask_post = slice(None)
    diff = np.abs(full_res.mass_fractions[mask_post] - Y_red_full[mask_post])
    pen_species = float(np.sum(weights_arr * diff.mean(axis=0)))

    keep_cnt = int(sel.sum())
    size_frac = keep_cnt / len(selection)
    size_pen = 3.0 * max(0.0, size_frac - 0.4)
    if target_species is not None:
        qpen = ((keep_cnt - target_species) / len(selection)) ** 2
        size_pen += 3.0 * qpen

    if fitness_mode == "threshold":
        ratios = (
            err / max(tol_pv, 1e-12),
            delay_metric / max(tol_delay, 1e-12),
            tau_mis / max(tol_timescale, 1e-12),
            pen_species / max(tol_resid, 1e-12),
        )
        exceed = [max(0.0, r - 1.0) for r in ratios]
        if any(val > 0.0 for val in exceed):
            reason = "threshold_fail"
            fitness = -1e6 * sum(exceed) - size_pen
        else:
            margin = sum(1.0 - r for r in ratios)
            fitness = 100.0 + margin - size_pen - 0.1 * tau_mis
    else:
        fitness = -(1.0 * err + 12.0 * delay_metric + 1.0 * tau_mis + zeta * pen_species + size_pen)

    info = (
        float(err),
        float(delay_metric),
        float(size_pen),
        float(tau_mis),
        float(pen_species),
        float(ign_shift),
        reason,
        keep_cnt,
        float(delta_T),
        float(delta_Y),
        float(delay_red),
    )
    logger.info(
        "ΔT=%.3e ΔY=%.3e τred=%.3e metric=%.3e pv_err=%.3e τ_mis=%.3e pen=%.3e shift=%.3e fitness=%.3e",
        delta_T,
        delta_Y,
        delay_red,
        delay_metric,
        err,
        tau_mis,
        pen_species,
        ign_shift,
        fitness,
    )

    return float(fitness), *info


def evaluate_selection_multi(
    selection,
    base_mech,
    weights,
    critical_idxs,
    runner,
    training_data: Sequence[dict],
    *,
    fitness_mode: str = "threshold",
    tol_pv: float = 0.05,
    tol_delay: float = 0.05,
    tol_timescale: float = 0.05,
    tol_resid: float = 0.05,
    target_species: int | None = None,
    zeta: float = 20.0,
    details: list[dict] | None = None,
):
    if critical_idxs and (selection.sum() < max(4, len(critical_idxs)) or np.any(selection[critical_idxs] == 0)):
        info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, "missing_critical", int(selection.sum()), 0.0, 0.0, 0.0)
        return -1e6, *info

    mech = Mechanism(base_mech.file_path)
    keep = [base_mech.species_names[i] for i, bit in enumerate(selection) if bit]
    remove = [s for s in mech.species_names if s not in keep]
    if remove:
        mech.remove_species(remove)

    if not training_data:
        info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, "no_training_cases", int(selection.sum()), 0.0, 0.0, 0.0)
        return -1e6, *info

    case_results: list[dict] = []
    tol_vec = (
        max(tol_pv, 1e-12),
        max(tol_delay, 1e-12),
        max(tol_timescale, 1e-12),
        max(tol_resid, 1e-12),
    )

    for entry in training_data:
        weight = float(entry.get("weight", 1.0))
        full_case = entry.get("full")
        if full_case is None:
            continue
        Y_case = entry.get("Y0")
        T_case = float(entry.get("T0", 0.0))
        p_case = float(entry.get("p", 0.0))
        time_grid = getattr(full_case, "time", None)
        steps_case = int(entry.get("steps", max(len(time_grid) - 1, 1))) if time_grid is not None else max(int(entry.get("steps", 0)), 1)

        try:
            red_case = runner(
                mech.solution,
                T_case,
                p_case,
                Y_case,
                float(time_grid[-1]) if time_grid is not None else 1.0,
                nsteps=steps_case,
                use_mole=False,
                time_grid=time_grid,
            )
        except RuntimeError as e:
            reason = f"sim_failed:{entry.get('id', '?')}:{e}" if "no_reaction" not in str(e) else "no_reaction"
            info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, reason, int(selection.sum()), 0.0, 0.0, 0.0)
            return -1e6, *info
        except Exception as e:  # pragma: no cover - defensive
            info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, f"sim_failed:{type(e).__name__}", int(selection.sum()), 0.0, 0.0, 0.0)
            return -1e6, *info

        metrics = _compute_case_metrics(
            full_case,
            red_case,
            base_mech.species_names,
            mech.species_names,
            weights,
            tol_pv,
            tol_delay,
            tol_timescale,
            tol_resid,
        )

        delta_T = float(red_case.temperature[-1] - red_case.temperature[0])
        delta_Y = float(
            np.max(np.abs(red_case.mass_fractions[-1] - red_case.mass_fractions[0]))
        )
        ratios = (
            metrics["pv_err"] / tol_vec[0],
            metrics["delay_metric"] / tol_vec[1],
            metrics["tau_mis"] / tol_vec[2],
            metrics["pen_species"] / tol_vec[3],
        )

        case_results.append(
            {
                "id": entry.get("id", "case"),
                "application": entry.get("application", ""),
                "weight": weight,
                "metrics": metrics,
                "ratios": ratios,
                "delta_T": delta_T,
                "delta_Y": delta_Y,
                "delay_red": metrics.get("delay_red", 0.0),
            }
        )

        if details is not None:
            details.append(
                {
                    "case_id": entry.get("id", "case"),
                    "application": entry.get("application", ""),
                    "weight": weight,
                    "pv_err": float(metrics["pv_err"]),
                    "delay_metric": float(metrics["delay_metric"]),
                    "tau_mis": float(metrics["tau_mis"]),
                    "pen_species": float(metrics["pen_species"]),
                    "ign_shift": float(metrics["ign_shift"]),
                    "passes": bool(metrics["passes"]),
                    "ratio_pv": float(ratios[0]),
                    "ratio_delay": float(ratios[1]),
                    "ratio_timescale": float(ratios[2]),
                    "ratio_resid": float(ratios[3]),
                }
            )

    if not case_results:
        info = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, "no_training_cases", int(selection.sum()), 0.0, 0.0, 0.0)
        return -1e6, *info

    weights_arr = np.array([max(cr["weight"], 0.0) for cr in case_results], dtype=float)
    weights_arr = np.where(weights_arr <= 0, 1.0, weights_arr)
    wsum = float(weights_arr.sum())
    if wsum <= 0.0:
        wsum = float(len(case_results))

    def _wavg(key: str) -> float:
        return float(
            sum(cr["weight"] * cr["metrics"][key] for cr in case_results) / wsum
        )

    pv_avg = _wavg("pv_err")
    delay_avg = _wavg("delay_metric")
    tau_avg = _wavg("tau_mis")
    pen_avg = _wavg("pen_species")
    ign_avg = _wavg("ign_shift")
    delta_T_avg = float(sum(cr["weight"] * cr["delta_T"] for cr in case_results) / wsum)
    delta_Y_avg = float(sum(cr["weight"] * cr["delta_Y"] for cr in case_results) / wsum)
    delay_red_avg = float(sum(cr["weight"] * cr["delay_red"] for cr in case_results) / wsum)

    ratios_avg = (
        pv_avg / tol_vec[0],
        delay_avg / tol_vec[1],
        tau_avg / tol_vec[2],
        pen_avg / tol_vec[3],
    )
    ratios_max = [max(cr["ratios"][i] for cr in case_results) for i in range(4)]
    ratios = tuple(max(avg, rmax) for avg, rmax in zip(ratios_avg, ratios_max))

    keep_cnt = int(selection.sum())
    size_frac = keep_cnt / len(selection)
    size_pen = 3.0 * max(0.0, size_frac - 0.4)
    if target_species is not None:
        qpen = ((keep_cnt - target_species) / len(selection)) ** 2
        size_pen += 3.0 * qpen

    fails = [cr["id"] for cr in case_results if not cr["metrics"]["passes"]]
    exceed = [max(0.0, r - 1.0) for r in ratios]
    reason = ""

    if fitness_mode == "threshold":
        if any(exceed) or fails:
            reason = "threshold_fail"
            if fails:
                reason += ":" + ",".join(map(str, fails))
            fitness = -1e6 * (sum(exceed) + (1.0 if fails else 0.0)) - size_pen
        else:
            margin = sum(1.0 - r for r in ratios)
            fitness = 100.0 + margin - size_pen - 0.1 * tau_avg
    else:
        fitness = -(pv_avg + 12.0 * delay_avg + tau_avg + zeta * pen_avg + size_pen)

    info = (
        float(pv_avg),
        float(delay_avg),
        float(size_pen),
        float(tau_avg),
        float(pen_avg),
        float(ign_avg),
        reason,
        keep_cnt,
        float(delta_T_avg),
        float(delta_Y_avg),
        float(delay_red_avg),
    )

    logger.info(
        "Multi-case fitness=%.3e pv=%.3e delay=%.3e tau=%.3e pen=%.3e size_pen=%.3e", 
        fitness,
        pv_avg,
        delay_avg,
        tau_avg,
        pen_avg,
        size_pen,
    )

    return float(fitness), *info


# -------------------------------
# GA driver
# -------------------------------

def run_ga_reduction(
    mech_path: str,
    Y0: Dict[str, float],
    tf: float,
    steps: int,
    T0: float,
    p0: float,
    log_times: bool,
    alpha: float,
    tf_short: float,
    steps_short: int,
    cache_labels: bool,
    isothermal: bool,
    min_species: int | None = None,
    max_species: int | None = None,
    target_species: int | None = None,
    generations: int = 60,
    population_size: int = 40,
    mutation_rate: float = 0.25,
    runner: Callable | None = None,
    fitness_mode: str = "standard",
    tol_pv: float = 0.05,
    tol_delay: float = 0.05,
    tol_timescale: float = 0.05,
    tol_resid: float = 0.05,
    training_data: Sequence[dict] | None = None,
    mode: str = "0d",
    out_dir: str = "results",
):
    mech = Mechanism(mech_path)

    runner_fn = runner or (run_isothermal_const_p if isothermal else run_constant_pressure)

    # Reference run (full window)
    full = runner_fn(
        mech.solution,
        T0,
        p0,
        Y0,
        tf,
        nsteps=steps,
        use_mole=False,
        log_times=log_times,
    )
    delay_full, slope_full = ignition_delay(full.time, full.temperature)
    if log_times:
        t0 = max(1e-12, 0.8 * delay_full)
        t1 = min(tf, 1.6 * delay_full)
        dense = np.linspace(t0, t1, 200)
        coarse = np.geomspace(1e-12, tf, steps)
        time_grid = np.unique(np.concatenate((coarse, dense)))
        time_grid = np.insert(time_grid, 0, 0.0)
        full = runner_fn(
            mech.solution,
            T0,
            p0,
            Y0,
            tf,
            nsteps=len(time_grid) - 1,
            use_mole=False,
            log_times=False,
            time_grid=time_grid,
        )
        delay_full, slope_full = ignition_delay(full.time, full.temperature)
    logger.info(
        "Reference ignition delay %.3e s, max dT/dt %.3e K/s",
        delay_full,
        slope_full,
    )
    if (full.temperature[-1] - full.temperature[0]) < 5.0 or (
        np.max(np.abs(full.mass_fractions[-1] - full.mass_fractions[0])) < 1e-6
    ):
        raise RuntimeError("Reference case looks inert (ΔT or ΔY too small). Check T0/φ/tf.")

    genome_len = len(mech.species_names)

    pv_full, tau_pv_full = pv_timescale(full.time, full.mass_fractions, mech.species_names)
    tau_spts_full = spts(full.time, full.mass_fractions)

    # PV weights for alignment
    weights = _pv_weights_for(mech)
    logger.info("Species weights (first 5): %s", list(zip(mech.species_names, weights))[:5])

    # --- Short-window baseline for LOO
    full_short = _baseline_short_run(mech, T0, p0, Y0, tf_short, steps_short, log_times, runner_fn)

    # Graph + labels
    G = build_species_graph(mech.solution)
    critical = [
        s
        for s in [
            "CH4",
            "O2",
            "N2",
            "H2O",
            "CO",
            "CO2",
            "OH",
            "H",
            "O",
            "HO2",
            "H2O2",
            "H2",
        ]
        if s in mech.species_names
    ]

    labels = _build_species_labels(
        mech,
        G,
        full_short,
        T0,
        p0,
        Y0,
        tf_short,
        steps_short,
        log_times,
        alpha,
        critical,
        cache_labels,
        runner_fn,
    )

    # GNN training (labels are already normalized 0..1)
    gnn_model = train_gnn(
        G,
        mech.solution,
        labels,
        epochs=200,
    )

    os.makedirs(out_dir, exist_ok=True)

    scores = predict_scores(
        model=gnn_model,
        G=G,
        solution=mech.solution,
        save_path=os.path.join(out_dir, "gnn_scores.csv"),
    )

    scores_arr = np.array([scores[s] for s in mech.species_names])
    std = float(scores_arr.std())
    logger.info(
        "GNN score stats min=%.3f mean=%.3f max=%.3f std=%.3e",
        float(scores_arr.min()),
        float(scores_arr.mean()),
        float(scores_arr.max()),
        std,
    )

    # Fallback if the GNN is degenerate
    if std < 1e-6:
        logger.warning("Using degree centrality as fallback for seeding.")
        seed_scores = np.array([G.degree(n) for n in mech.species_names], dtype=float)
    else:
        seed_scores = scores_arr

    order = np.argsort(seed_scores)

    # GA seeding & constraints
    critical_idxs = [mech.species_names.index(s) for s in critical]
    pop_size = population_size
    k = max(len(critical_idxs), int(0.2 * genome_len))

    seed = np.zeros(genome_len, dtype=int)
    seed[order[-k:]] = 1
    seed[critical_idxs] = 1

    init_pop = np.zeros((pop_size, genome_len), dtype=int)
    for i in range(pop_size):
        individual = seed.copy()
        flips = np.random.choice(genome_len, int(0.2 * genome_len), replace=False)
        individual[flips] ^= 1
        init_pop[i] = individual

    if mode == "1d" and training_data:
        eval_fn = lambda sel: evaluate_selection_multi(
            sel,
            mech,
            weights,
            critical_idxs,
            runner,
            training_data,
            fitness_mode=fitness_mode,
            tol_pv=tol_pv,
            tol_delay=tol_delay,
            tol_timescale=tol_timescale,
            tol_resid=tol_resid,
            target_species=target_species,
        )
    else:
        eval_fn = lambda sel: evaluate_selection(
            sel,
            mech,
            Y0,
            tf,
            full,
            weights,
            critical_idxs,
            T0,
            p0,
            len(full.time) - 1,
            log_times,
            runner,
            delay_full,
            tau_pv_full,
            tau_spts_full,
            target_species,
            fitness_mode,
            tol_pv,
            tol_delay,
            tol_timescale,
            tol_resid,
            mode,
        )

    ms = min_species or (len(critical_idxs) + 5)
    M = max_species or max(ms + 5, int(0.6 * genome_len))

    sel, hist, debug = run_ga(
        genome_len,
        eval_fn,
        GAOptions(
            population_size=pop_size,
            generations=generations,
            min_species=ms,
            max_species=M,
            mutation_rate=mutation_rate,
        ),
        return_history=True,
        initial_population=init_pop,
        return_debug=True,
        fixed_indices=critical_idxs,
    )

    # Debug drops per-gen
    os.makedirs(out_dir, exist_ok=True)
    for g, gen in enumerate(debug):
        best_g = max(gen, key=lambda x: x[-2])
        (
            pv_err,
            delay_diff,
            size_pen,
            tau_mis,
            pen_species,
            ign_shift,
            reason,
            keep_cnt,
            dT,
            dY,
            delay_red,
            fit,
            genome,
        ) = best_g
        logger.info(
            "gen %02d keep=%d ΔT=%.3e delay=%.3e pv_err=%.3e τ_mis=%.3e pen=%.3e shift=%.3e fitness=%.3e",
            g,
            keep_cnt,
            dT,
            delay_red,
            pv_err,
            tau_mis,
            pen_species,
            ign_shift,
            fit,
        )
        with open(os.path.join(out_dir, f"best_selection_gen{g:02d}.txt"), "w") as f:
            f.write(",".join(map(str, genome.tolist())))

    # Selection report
    kept = []
    dropped = []
    for i, s in enumerate(mech.species_names):
        if sel[i]:
            kept.append((s, labels.get(s, 0.0), scores.get(s, 0.0)))
        else:
            dropped.append((s, labels.get(s, 0.0), scores.get(s, 0.0)))
    kept.sort(key=lambda x: x[1], reverse=True)
    dropped.sort(key=lambda x: x[1], reverse=True)
    with open(os.path.join(out_dir, "selection_report.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "species", "label", "score"])
        for status, data in [("kept", kept[:20]), ("dropped", dropped[:20])]:
            for s, lab, sc in data:
                writer.writerow([status, s, lab, sc])

    with open("best_selection.txt", "w") as f:
        f.write(",".join(map(str, sel.tolist())))

    return ["GA"], [sel], [hist], debug, full, weights


def _load_envelopes(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Envelope specification '{path}' not found")
    with open(path) as f:
        return json.load(f)


def _entry_nominal(entry: dict | None) -> float | None:
    if not entry:
        return None
    if entry.get("nominal") is not None:
        return float(entry["nominal"])
    values = [entry.get("min"), entry.get("max")]
    vals = [float(v) for v in values if v is not None]
    if vals:
        return float(np.mean(vals))
    return None


def _entry_min(entry: dict | None) -> float | None:
    if not entry:
        return None
    if entry.get("min") is not None:
        return float(entry["min"])
    return _entry_nominal(entry)


def _entry_max(entry: dict | None) -> float | None:
    if not entry:
        return None
    if entry.get("max") is not None:
        return float(entry["max"])
    return _entry_nominal(entry)


def _two_point_cases(name: str, env: dict) -> list[dict]:
    phi_min = _entry_min(env.get("phi"))
    phi_max = _entry_max(env.get("phi"))
    T_min = _entry_min(env.get("T0_K"))
    T_max = _entry_max(env.get("T0_K"))
    p_min = _entry_min(env.get("p_bar"))
    p_max = _entry_max(env.get("p_bar"))
    p_nom = _entry_nominal(env.get("p_bar")) or 1.0
    diluents = env.get("diluents", [])
    ratio = env.get("CO2_CH4_ratio") or {}
    ratio_nom = float(ratio.get("nominal")) if isinstance(ratio, dict) and ratio.get("nominal") is not None else None

    def _pa(val: float | None) -> float:
        return float((val if val is not None else p_nom) * 1e5)

    phi_low = float(phi_min if phi_min is not None else (phi_max or 1.0))
    phi_high = float(phi_max if phi_max is not None else (phi_min or 1.0))
    T_high = float(T_max if T_max is not None else (T_min or 300.0))
    T_low = float(T_min if T_min is not None else (T_max or 300.0))

    cases = [
        {
            "id": f"{name.lower()}_low",
            "application": name,
            "phi": phi_low,
            "T0": T_high,
            "p": _pa(p_min),
            "diluents": diluents,
            "co2_ch4_ratio": ratio_nom,
            "label": "min_phi_max_T",
        },
        {
            "id": f"{name.lower()}_high",
            "application": name,
            "phi": phi_high,
            "T0": T_low,
            "p": _pa(p_max),
            "diluents": diluents,
            "co2_ch4_ratio": ratio_nom,
            "label": "max_phi_min_T",
        },
    ]
    return cases


def _load_training_cases_from_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training case file '{path}' not found")

    ext = os.path.splitext(path)[1].lower()
    entries: list[dict]
    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            entries = list(data.get("cases", []))
        elif isinstance(data, list):
            entries = list(data)
        else:
            raise ValueError("Training case JSON must be a list or contain a 'cases' list")
    else:
        entries = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)

    cases: list[dict] = []

    def _parse_float(val, fallback=None):
        if val is None or val == "":
            return fallback
        try:
            return float(val)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid float value '{val}' in training case file") from exc

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        case = dict(entry)
        case_id = case.get("id") or f"train_case_{idx:02d}"
        application = str(case.get("application", "POX"))
        phi = _parse_float(case.get("phi"))
        T0 = _parse_float(case.get("T0"))
        p = case.get("p")
        if p is None:
            p = case.get("p_pa")
        if p is None and case.get("p_bar") is not None:
            p = _parse_float(case.get("p_bar"))
            if p is not None:
                p *= 1e5
        else:
            p = _parse_float(p)

        if phi is None or T0 is None or p is None:
            raise ValueError(f"Training case '{case_id}' missing phi/T0/p")

        dil = case.get("diluents", [])
        if isinstance(dil, str):
            try:
                dil = json.loads(dil)
            except json.JSONDecodeError:
                dil = []
        if not isinstance(dil, list):
            dil = []

        ratio = case.get("co2_ch4_ratio")
        ratio_val = _parse_float(ratio) if ratio is not None else None

        use_seed_raw = case.get("use_seed", False)
        if isinstance(use_seed_raw, str):
            use_seed = use_seed_raw.strip().lower() in {"1", "true", "yes"}
        else:
            use_seed = bool(use_seed_raw)

        cases.append(
            {
                "id": str(case_id),
                "application": application,
                "phi": float(phi),
                "T0": float(T0),
                "p": float(p),
                "diluents": dil,
                "co2_ch4_ratio": ratio_val,
                "use_seed": use_seed,
            }
        )

    return cases


def _parse_train_weights(spec: str | None, cases: list[dict]) -> list[float]:
    if not cases:
        return []
    if spec is None or spec.strip().lower() in {"", "auto"}:
        return [1.0] * len(cases)

    tokens = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        return [1.0] * len(cases)

    if all("=" in tok for tok in tokens):
        mapping = {}
        for tok in tokens:
            key, val = tok.split("=", 1)
            mapping[key.strip().lower()] = float(val)

        def _group(case: dict) -> str:
            app = str(case.get("application", "")).lower()
            if app in {"hp_pox", "hp-pox"}:
                return "hp_pox"
            if app in {"co2_recycle", "co2-recycle", "co2"}:
                return "co2"
            if app == "plasma":
                return "plasma"
            return "pox"

        return [float(mapping.get(_group(case), 1.0)) for case in cases]

    if len(tokens) != len(cases):
        raise ValueError(
            "Number of training weights must match training cases or provide named groups"
        )
    return [float(tok) for tok in tokens]


# at top of testing/pipeline.py (imports section), make sure this exists:
# from mechanism.mix import methane_air_mole_fractions, methane_oxygen_mole_fractions

def _compose_feed(
    sol: ct.Solution,
    phi: float,
    base_diluent: str,
    diluent_specs: list[dict],
    *,
    radical_seed: Dict[str, float] | None = None,
    co2_ratio: float | None = None,
) -> Dict[str, float]:
    """
    Build inlet *mole-fraction* map, then convert to mass fractions for Cantera.
    - base_diluent:
        "OXY"  -> O2-blown (no carrier N2), uses methane_oxygen_mole_fractions(phi)
        else   -> methane_air_mole_fractions(phi, diluent=base_diluent)  (e.g., N2/Ar/He)
    - diluent_specs: [{"species":"H2O","fraction":{"nominal":0.30}}, ...]  fractions are *mole* basis
    - radical_seed:  {"H":0.005, "OH":0.002}   (mole fractions)
    - co2_ratio: enforce CO2 >= co2_ratio * CH4 (mole basis), if provided
    """

    # ---- 1) Choose baseline (O2-blown vs air-like) ----
    try:
        if str(base_diluent).upper() == "OXY":
            mix = methane_oxygen_mole_fractions(phi)  # CH4 + O2 only (no N2 carrier)
        else:
            mix = methane_air_mole_fractions(phi, diluent=base_diluent)  # CH4 + O2 + (carrier)
    except Exception as e:
        logger.warning("Baseline mixing failed with '%s': %s. Falling back to methane_air (N2).",
                       base_diluent, e)
        mix = methane_air_mole_fractions(phi, diluent="N2")

    # Sanity: keep only species present in mechanism
    mech_species = set(sol.species_names)
    mix = {sp: float(x) for sp, x in mix.items() if sp in mech_species and x > 0.0}

    # ---- 2) Accumulate envelope diluents on mole basis ----
    dil_map: Dict[str, float] = {}
    dil_total = 0.0
    for item in (diluent_specs or []):
        if not isinstance(item, dict):
            continue
        species = str(item.get("species", "")).strip()
        frac = item.get("fraction", {})
        val = None
        if isinstance(frac, dict):
            val = frac.get("nominal", None)
        if species and (val is not None):
            try:
                v = float(val)
            except Exception:
                continue
            if v <= 0.0:
                continue
            if species not in mech_species:
                logger.warning("Diluent '%s' not in mechanism; skipping.", species)
                continue
            dil_map[species] = dil_map.get(species, 0.0) + v
            dil_total += v

    # ---- 3) Radical seeds (optional) on mole basis ----
    seed_map = {}
    if radical_seed:
        for sp, v in radical_seed.items():
            try:
                vv = float(v)
            except Exception:
                continue
            if vv <= 0.0:
                continue
            if sp not in mech_species:
                logger.warning("Radical seed '%s' not in mechanism; skipping.", sp)
                continue
            seed_map[sp] = seed_map.get(sp, 0.0) + vv
    seed_total = float(sum(seed_map.values()))

    # ---- 4) Prevent overfill (diluent + seed >= 1). Scale down proportionally. ----
    over = dil_total + seed_total
    if over >= 0.999:
        if over <= 0.0:
            scale = 1.0
        else:
            scale = 0.999 / over
        if scale < 1.0:
            logger.warning("Diluent+seed (%.3f) exceeds unity; scaling both by %.3f.", over, scale)
            for sp in list(dil_map.keys()):
                dil_map[sp] *= scale
            for sp in list(seed_map.keys()):
                seed_map[sp] *= scale
            dil_total = sum(dil_map.values())
            seed_total = sum(seed_map.values())

    # ---- 5) Compose final mole fractions: base gets the remainder ----
    base_fraction = max(1e-6, 1.0 - dil_total - seed_total)
    scaled: Dict[str, float] = {}
    for sp, val in mix.items():
        scaled[sp] = val * base_fraction

    # add diluents
    for sp, frac in dil_map.items():
        scaled[sp] = scaled.get(sp, 0.0) + frac
    # add radicals
    for sp, frac in seed_map.items():
        scaled[sp] = scaled.get(sp, 0.0) + frac

    # ---- 6) Optional CO2/CH4 ratio enforcement (mole basis) ----
    if (co2_ratio is not None) and (co2_ratio > 0.0) and ("CH4" in scaled):
        ch4 = max(scaled.get("CH4", 0.0), 0.0)
        target = co2_ratio * ch4
        if target > 0.0:
            scaled["CO2"] = max(scaled.get("CO2", 0.0), target)

    # ---- 7) Clean-up, normalize mole fractions ----
    # remove tiny negatives / numerical noise
    for sp, v in list(scaled.items()):
        if not np.isfinite(v) or v <= 0.0:
            scaled.pop(sp, None)
    total = float(sum(scaled.values()))
    if total <= 0.0:
        raise ValueError("Mixture fractions sum to zero after composition.")
    scaled = {k: float(v) / total for k, v in scaled.items()}

    # Final guard: ensure fuel & oxidizer exist in some amount
    if "CH4" not in scaled:
        logger.warning("No CH4 present after composition; injecting a tiny fuel tracer.")
        scaled["CH4"] = 1e-6
    if "O2" not in scaled and any(sp in scaled for sp in ("NO2", "N2O", "O", "OH")) is False:
        # only warn when truly zero oxidizer-ish species
        logger.warning("No O2 (or oxidizer radical) present; adding tiny O2 tracer.")
        scaled["O2"] = scaled.get("O2", 0.0) + 1e-6
        # renormalize
        s = float(sum(scaled.values()))
        scaled = {k: v / s for k, v in scaled.items()}

    # ---- 8) Convert mole -> mass fractions for Cantera ----
    return mole_to_mass_fractions(sol, scaled)


def _compute_case_metrics(
    full_res,
    red_res,
    species_full: Sequence[str],
    species_red: Sequence[str],
    weights: Sequence[float],
    tol_pv: float,
    tol_delay: float,
    tol_timescale: float,
    tol_resid: float,
) -> Dict[str, float]:
    weights_arr = np.asarray(weights, dtype=float)
    map_full = {s: i for i, s in enumerate(species_full)}
    map_red = {s: i for i, s in enumerate(species_red)}

    Y_red_interp = _interp_to(full_res.time, red_res.time, red_res.mass_fractions)

    pv_err = pv_error_aligned(
        full_res.mass_fractions,
        Y_red_interp,
        species_full,
        species_red,
        weights_arr,
    )
    delay_full, _ = ignition_delay(full_res.time, full_res.temperature)
    delay_red, _ = ignition_delay(red_res.time, red_res.temperature)
    delay_ratio = abs(delay_red - delay_full) / max(delay_full, 1e-12)

    x_full = getattr(full_res, "x", None)
    x_red = getattr(red_res, "x", None)
    ign_shift = 0.0
    delay_metric = delay_ratio
    if x_full is not None and x_red is not None:
        x_full_arr = np.asarray(x_full, dtype=float)
        x_red_arr = np.asarray(x_red, dtype=float)
        if x_full_arr.size > 0 and x_red_arr.size > 0:
            if x_full_arr.size >= 2:
                x_full_arr = np.maximum.accumulate(x_full_arr)
            if x_red_arr.size >= 2:
                x_red_arr = np.maximum.accumulate(x_red_arr)
            try:
                x_ign_full = float(np.interp(delay_full, full_res.time, x_full_arr))
                x_ign_red = float(np.interp(delay_red, red_res.time, x_red_arr))
                ign_shift = abs(x_ign_red - x_ign_full) / max(x_ign_full, 1e-12)
                delay_metric = 0.5 * (delay_ratio + ign_shift)
            except Exception:
                ign_shift = 0.0
                delay_metric = delay_ratio

    _, tau_pv_full = pv_timescale(full_res.time, full_res.mass_fractions, species_full)
    _, tau_pv_red = pv_timescale(red_res.time, red_res.mass_fractions, species_red)
    tau_spts_full = spts(full_res.time, full_res.mass_fractions)
    tau_spts_red = spts(red_res.time, red_res.mass_fractions)

    log_full_pv = np.log10(tau_pv_full + 1e-30)
    log_red_pv = _interp_to(full_res.time, red_res.time, np.log10(tau_pv_red + 1e-30))
    log_full_spts = np.log10(tau_spts_full + 1e-30)
    log_red_spts = _interp_to(full_res.time, red_res.time, np.log10(tau_spts_red + 1e-30))

    tau_mis = _rms_norm(log_full_pv - log_red_pv) + _rms_norm(log_full_spts - log_red_spts)

    mask = full_res.time > delay_full
    if x_full is not None:
        x_full_arr = np.asarray(x_full, dtype=float)
        if x_full_arr.size >= 2:
            x_mono = np.maximum.accumulate(x_full_arr)
        else:
            x_mono = x_full_arr
        try:
            x_ign_full = float(np.interp(delay_full, full_res.time, x_mono))
        except Exception:
            x_ign_full = x_mono[x_mono.size // 10] if x_mono.size else 0.0
        if x_mono.size:
            L = float(x_mono[-1])
        else:
            L = 1.0
        upper = min(L, x_ign_full + 0.3 * L)
        alt = (x_mono >= x_ign_full) & (x_mono <= upper)
        if np.any(alt):
            mask = alt

    if isinstance(mask, np.ndarray) and not np.any(mask):
        mask = full_res.time > delay_full
        if isinstance(mask, np.ndarray) and not np.any(mask):
            mask = slice(None)

    Y_red_full = np.zeros_like(full_res.mass_fractions)
    for s, idx_full in map_full.items():
        idx_red = map_red.get(s)
        if idx_red is not None:
            Y_red_full[:, idx_full] = Y_red_interp[:, idx_red]

    diff = np.abs(full_res.mass_fractions[mask] - Y_red_full[mask])
    pen_species = float(np.sum(weights_arr * diff.mean(axis=0)))

    ratios = (
        pv_err / max(tol_pv, 1e-12),
        delay_metric / max(tol_delay, 1e-12),
        tau_mis / max(tol_timescale, 1e-12),
        pen_species / max(tol_resid, 1e-12),
    )
    passes = all(r <= 1.0 for r in ratios)
    exceed = [max(0.0, r - 1.0) for r in ratios]
    if passes:
        margin = sum(1.0 - r for r in ratios)
        score = 100.0 + margin - 0.1 * tau_mis
    else:
        score = -1e6 * sum(exceed)

    return {
        "pv_err": float(pv_err),
        "delay_metric": float(delay_metric),
        "tau_mis": float(tau_mis),
        "pen_species": float(pen_species),
        "ign_shift": float(ign_shift),
        "delay_full": float(delay_full),
        "delay_red": float(delay_red),
        "passes": passes,
        "score": float(score),
    }


# -------------------------------
# Full pipeline entry point
# -------------------------------

def _full_pipeline_batch(
    mech_path: str,
    out_dir: str,
    steps: int = 1000,
    tf: float = 0.5,
    phi: float | None = None,
    preset: str = "methane_air",
    T0: float | None = None,
    p0: float | None = None,
    log_times: bool = False,
    alpha: float = 0.8,
    tf_short: float | None = None,
    steps_short: int | None = None,
    cache_labels: bool = True,
    isothermal: bool = False,
    min_species: int | None = None,
    max_species: int | None = None,
    target_species: int | None = None,
    generations: int = 60,
    population: int = 40,
    mutation: float = 0.25,
    focus: str = "auto",
    focus_window: Tuple[float, float] | None = None,
    report_grid: str | None = None,
):
    mech = Mechanism(mech_path)

    # Mixture preset
    if preset == "methane_air":
        phi = phi or 1.0
        x0 = methane_air_mole_fractions(phi)
        Y0 = mole_to_mass_fractions(mech.solution, x0)
        T0 = T0 or 1500.0
        p0 = p0 or ct.one_atm
    elif preset in HR_PRESETS:
        T0, p0, Y0 = HR_PRESETS[preset](mech.solution)
    else:
        raise NotImplementedError(preset)

    tf = min(tf, 1.0)

    # Defaults for short LOO runs
    if tf_short is None:
        tf_short = 0.25 * tf
    if steps_short is None:
        steps_short = max(50, int(0.25 * steps))

    use_iso = isothermal and preset in HR_PRESETS
    runner = run_isothermal_const_p if use_iso else run_constant_pressure

    names, sols, hists, debug, full, weights = run_ga_reduction(
        mech_path,
        Y0,
        tf,
        steps,
        T0,
        p0,
        log_times,
        alpha,
        tf_short,
        steps_short,
        cache_labels,
        use_iso,
        min_species=min_species,
        max_species=max_species,
        target_species=target_species,
        generations=generations,
        population_size=population,
        mutation_rate=mutation,
        runner=runner,
        fitness_mode="standard",
        mode="0d",
        out_dir=out_dir,
    )

    os.makedirs(out_dir, exist_ok=True)

    # GA fitness history
    with open(os.path.join(out_dir, "ga_fitness.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "fitness"])
        for i, val in enumerate(hists[0]):
            writer.writerow([i, val])

    # Per-individual debug
    with open(os.path.join(out_dir, "debug_fitness.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "individual",
            "pv_error",
            "delay_diff",
            "size_penalty",
            "tau_mismatch",
            "pen_species",
            "ignition_shift",
            "reason",
            "keep_count",
            "delta_T",
            "delta_Y",
            "delay_red",
            "fitness",
        ])
        for g, gen in enumerate(debug):
            for i, d in enumerate(gen):
                pv_err, delay_diff, size_pen, tau_mis, pen_species, ign_shift, reason, keep_cnt, dT, dY, delay_red, fit, genome = d
                writer.writerow(
                    [g, i, pv_err, delay_diff, size_pen, tau_mis, pen_species, ign_shift, reason, keep_cnt, dT, dY, delay_red, fit]
                )

        # Build reduced mechanism from best selection
    best_sel = sols[0]
    keep = [mech.species_names[i] for i, bit in enumerate(best_sel) if bit]

    red_mech = Mechanism(mech_path)
    red_mech.remove_species([s for s in red_mech.species_names if s not in keep])

    with open(os.path.join(out_dir, "reduced_species.txt"), "w") as f:
        for s in keep:
            f.write(s + "\n")

    # Re-run reduced on same window/grid choice
    red = runner(
        red_mech.solution,
        T0,
        p0,
        Y0,
        tf,
        nsteps=len(full.time) - 1,
        use_mole=False,
        log_times=log_times,
        time_grid=full.time,
    )

    # Informative species (like paper)
    key_species = [s for s in ["O2", "CO2", "H2O", "CO", "CH4", "OH"] if s in mech.species_names]

    # --- ignition delays (both)
    delay_full, _ = ignition_delay(full.time, full.temperature)
    delay_red,  _ = ignition_delay(red.time,  red.temperature)

    # Residuals per species after ignition
    mask = full.time > delay_full
    map_full = {s: i for i, s in enumerate(mech.species_names)}
    map_red = {s: i for i, s in enumerate(red_mech.species_names)}
    red_interp_full = _interp_to(full.time, red.time, red.mass_fractions)
    with open(os.path.join(out_dir, "residual_species.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "mean_res", "max_res"])
        for s in mech.species_names:
            if s not in map_red:
                continue
            yf = full.mass_fractions[mask, map_full[s]]
            yr = red_interp_full[mask, map_red[s]]
            diff = np.abs(yf - yr)
            writer.writerow([s, float(diff.mean()), float(diff.max())])

    if focus == "auto":
        fw = (0.8 * delay_full, 1.6 * delay_full)
    elif focus == "window" and focus_window is not None:
        fw = focus_window
    else:
        fw = None

    plot_species_profiles(
        full.time,
        full.mass_fractions,
        mech.species_names,
        red.time,
        red.mass_fractions,
        red_mech.species_names,
        key_species,
        os.path.join(out_dir, "profiles"),
        tau_full=delay_full,
        tau_red=delay_red,
        focus=focus,
        focus_window=fw,
    )

    plot_species_residuals(
        full.time,
        full.mass_fractions,
        red.time,
        red.mass_fractions,
        mech.species_names,
        red_mech.species_names,
        key_species,
        os.path.join(out_dir, "profiles_residual"),
        tau_full=delay_full,
        focus=focus,
        focus_window=fw,
    )

    # 2) Ignition delay bars
    plot_ignition_delays(
        delays=[delay_full, delay_red],
        labels=["Full", "Reduced"],
        out_base=os.path.join(out_dir, "ignition_delay"),
    )

    # 3) GA convergence
    plot_convergence(hists, names, os.path.join(out_dir, "convergence"))

    # 4) PV error (aligned) + PV overlay
    pv_err = pv_error_aligned(
        full.mass_fractions,
        red_interp_full,
        mech.species_names,
        red_mech.species_names,
        weights,
    )
    with open(os.path.join(out_dir, "pv_error.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pv_error"])
        writer.writerow([pv_err])

    # PV overlay (compute BEFORE plotting)
    map_full = {s: i for i, s in enumerate(mech.species_names)}
    map_red = {s: i for i, s in enumerate(red_mech.species_names)}
    use = [s for s in PV_SPECIES_DEFAULT if s in map_full and s in map_red]
    if use:
        w = np.array([weights[map_full[s]] for s in use])
        pv_full = (
            full.mass_fractions[:, [map_full[s] for s in use]] * w
        ).sum(axis=1)
        pv_red = (
            red.mass_fractions[:, [map_red[s] for s in use]] * w
        ).sum(axis=1)
    else:
        pv_full = np.zeros_like(full.time)
        pv_red = np.zeros_like(red.time)

    plot_progress_variable(
        full.time,
        pv_full,
        red.time,
        pv_red,
        os.path.join(out_dir, "pv_overlay"),
        tau=delay_full,
        focus=focus,
        focus_window=fw,
    )

    # 5) Time-scales overlay (PVTS / SPTS)
    _, tau_pv_full = pv_timescale(full.time, full.mass_fractions, mech.species_names)
    _, tau_pv_red  = pv_timescale(red.time,  red.mass_fractions,  red_mech.species_names)
    tau_spts_full  = spts(full.time, full.mass_fractions)
    tau_spts_red   = spts(red.time,  red.mass_fractions)

    plot_timescales(
        full.time,
        tau_pv_full,
        tau_spts_full,
        red.time,
        tau_pv_red,
        tau_spts_red,
        os.path.join(out_dir, "timescales"),
    )

    with open(os.path.join(out_dir, "timescales.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time","tau_pv_full","tau_pv_red","tau_spts_full","tau_spts_red"])
        for t, tpvf, tpvr, tsf, tsr in zip(
            full.time, tau_pv_full, tau_pv_red, tau_spts_full, tau_spts_red
        ):
            writer.writerow([t, tpvf, tpvr, tsf, tsr])

    # Robustness grid evaluation
    if report_grid:
        out_path = os.path.join(out_dir, "robustness.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "phi",
                    "T0",
                    "p0",
                    "feasible",
                    "pv_err",
                    "delay_diff",
                    "tau_mis",
                    "post_ign_resid",
                    "ignition_shift",
                    "kept_species",
                ]
            )
            for triple in report_grid.split(","):
                phi_str, T0_str, p0_str = triple.split(":")
                phi = float(phi_str)
                T0g = float(T0_str)
                p0g = float(p0_str)

                if preset == "methane_air":
                    x0 = methane_air_mole_fractions(phi)
                    Y0g = mole_to_mass_fractions(mech.solution, x0)
                else:
                    T0g, p0g, Y0g = HR_PRESETS[preset](mech.solution)

                (
                    fit,
                    pv_e,
                    delay_d,
                    size_p,
                    tau_m,
                    pen,
                    ign_shift,
                    reason,
                    keep_cnt,
                    dT,
                    dY,
                    delay_r,
                ) = evaluate_selection(
                    sols[0],
                    mech,
                    Y0g,
                    tf,
                    full,
                    weights,
                    [],
                    T0g,
                    p0g,
                    steps,
                    log_times,
                    runner,
                    delay_full,
                    tau_pv_full,
                    tau_spts_full,
                    None,
                )
                feasible = np.isfinite(fit)
                writer.writerow(
                    [phi, T0g, p0g, feasible, pv_e, delay_d, tau_m, pen, ign_shift, int(sols[0].sum())]
                )

    # --- Console summary
    print(f"\nSummary:\n{'Mechanism':>10} {'Species':>8} {'Reactions':>10} {'Delay[s]':>12} {'PV err':>8}")
    print(f"{'Full':>10} {len(mech.species_names):8d} {len(mech.reactions()):10d} {delay_full:12.3e} {0.0:8.3f}")
    print(f"{'Reduced':>10} {len(red_mech.species_names):8d} {len(red_mech.reactions()):10d} {delay_red:12.3e} {pv_err:8.3f}")
    if pv_err < 0.05:
        print("Summary:\n"
              f"  Species: {len(mech.species_names)} -> {len(red_mech.species_names)}\n"
              f"  Reactions: {len(mech.reactions())} -> {len(red_mech.reactions())}\n"
              f"  Delay_full/red: {delay_full:.3e} / {delay_red:.3e} s\n"
              f"  PV_error: {pv_err*100:.1f}%")


def _full_pipeline_pfr(
    mech_path: str,
    out_dir: str,
    steps: int,
    phi: float | None,
    T0: float | None,
    p0: float | None,
    log_times: bool,
    alpha: float,
    tf_short: float | None,
    steps_short: int | None,
    cache_labels: bool,
    min_species: int | None,
    max_species: int | None,
    target_species: int | None,
    generations: int,
    population: int,
    mutation: float,
    fitness_mode: str,
    tol_pv: float,
    tol_delay: float,
    tol_timescale: float,
    tol_resid: float,
    diluent: str,
    L: float,
    D: float,
    mdot: float,
    U: float,
    Tw: float | None,
    plasma_length: float,
    T_plasma_out: float | None,
    radical_seed: Dict[str, float] | None,
    envelopes_path: str,
    train_cases: str = "auto",
    train_weights: str | None = None,
    topn_species: int = 6,
):
    mech = Mechanism(mech_path)
    envelopes = _load_envelopes(envelopes_path)
    pox_env = envelopes.get("POX", {})

    phi_base = float(phi if phi is not None else (_entry_nominal(pox_env.get("phi")) or 1.5))
    T0_base = float(T0 if T0 is not None else (_entry_nominal(pox_env.get("T0_K")) or 520.0))
    p0_base = float(p0 if p0 is not None else ((_entry_nominal(pox_env.get("p_bar")) or 5.0) * 1e5))

    seed_dict = radical_seed or {}
    Y0 = _compose_feed(
        mech.solution,
        phi_base,
        diluent,
        pox_env.get("diluents", []),
        radical_seed=seed_dict if seed_dict else None,
    )

    config = PFRConfig(
        length=L,
        diameter=D,
        mass_flow=mdot,
        pressure=p0_base,
        n_points=max(steps + 1, 300),
        heat_transfer_coeff=U,
        wall_temperature=Tw,
        enable_heat_loss=U > 0.0,
        plasma_length=plasma_length,
        plasma_temperature=T_plasma_out,
    )
    runner = PFRRunner(config, base_points=max(steps + 1, 300))

    base_res = runner(
        mech.solution,
        T0_base,
        p0_base,
        Y0,
        tf=1.0,
        nsteps=steps,
        use_mole=False,
    )
    tf_total = float(base_res.time[-1])
    steps_effective = len(base_res.time) - 1
    if tf_short is None:
        tf_short = 0.4 * tf_total
    if steps_short is None:
        steps_short = max(50, max(steps_effective // 2, 10))
    config.max_residence_time = max(config.max_residence_time or 0.0, 1.5 * tf_total)

    if isinstance(train_cases, str) and train_cases.lower() == "auto":
        ratio = pox_env.get("CO2_CH4_ratio") or {}
        ratio_nom = None
        if isinstance(ratio, dict) and ratio.get("nominal") is not None:
            try:
                ratio_nom = float(ratio.get("nominal"))
            except Exception:
                ratio_nom = None
        auto_cases: list[dict] = [
            {
                "id": "pox_nominal",
                "application": "POX",
                "phi": phi_base,
                "T0": T0_base,
                "p": p0_base,
                "diluents": pox_env.get("diluents", []),
                "co2_ch4_ratio": ratio_nom,
                "use_seed": False,
            }
        ]
        for name in ("POX", "HP_POX", "CO2_recycle", "PLASMA"):
            env = envelopes.get(name)
            if not env:
                continue
            for case in _two_point_cases(name, env):
                spec = dict(case)
                spec.setdefault("diluents", env.get("diluents", []))
                spec.setdefault("use_seed", name == "PLASMA")
                auto_cases.append(spec)
        train_specs = auto_cases
    else:
        train_specs = _load_training_cases_from_file(train_cases)

    if not train_specs:
        train_specs = [
            {
                "id": "pox_nominal",
                "application": "POX",
                "phi": phi_base,
                "T0": T0_base,
                "p": p0_base,
                "diluents": pox_env.get("diluents", []),
                "co2_ch4_ratio": None,
                "use_seed": False,
            }
        ]

    weight_specs = _parse_train_weights(train_weights, train_specs)
    if len(weight_specs) != len(train_specs):
        raise ValueError("Training weights must match number of training cases")

    training_data: list[dict] = []
    training_lookup: dict[str, dict] = {}

    for spec, weight in zip(train_specs, weight_specs):
        case_id = str(spec.get("id", f"case_{len(training_data)}"))
        use_seed = bool(spec.get("use_seed", False))
        seed = seed_dict if use_seed and seed_dict else None

        if (
            abs(float(spec.get("phi", phi_base)) - phi_base) < 1e-9
            and abs(float(spec.get("T0", T0_base)) - T0_base) < 1e-6
            and abs(float(spec.get("p", p0_base)) - p0_base) < 1.0
        ):
            Y_case = dict(Y0)
            full_case = base_res
        else:
            Y_case = _compose_feed(
                mech.solution,
                float(spec.get("phi", phi_base)),
                diluent,
                spec.get("diluents", []),
                radical_seed=seed,
                co2_ratio=spec.get("co2_ch4_ratio"),
            )
            full_case = runner(
                mech.solution,
                float(spec.get("T0", T0_base)),
                float(spec.get("p", p0_base)),
                Y_case,
                tf_total,
                nsteps=steps_effective,
                use_mole=False,
            )

        delta_T_full = float(full_case.temperature[-1] - full_case.temperature[0])
        delta_Y_full = float(
            np.max(np.abs(full_case.mass_fractions[-1] - full_case.mass_fractions[0]))
        )
        if delta_T_full < 5.0 and delta_Y_full < 1e-6:
            logger.warning("Skipping inert training case %s", case_id)
            continue

        entry = {
            "id": case_id,
            "application": spec.get("application", ""),
            "weight": float(weight),
            "Y0": dict(Y_case),
            "T0": float(spec.get("T0", T0_base)),
            "p": float(spec.get("p", p0_base)),
            "full": full_case,
            "steps": len(full_case.time) - 1,
        }
        training_data.append(entry)
        training_lookup[case_id] = entry

    if not training_data:
        raise RuntimeError("No valid training cases available for 1D reduction")

    names, sols, hists, debug, full, weights = run_ga_reduction(
        mech_path,
        Y0,
        tf_total,
        steps_effective,
        T0_base,
        p0_base,
        log_times,
        alpha,
        tf_short,
        steps_short,
        cache_labels,
        False,
        min_species=min_species,
        max_species=max_species,
        target_species=target_species,
        generations=generations,
        population_size=population,
        mutation_rate=mutation,
        runner=runner,
        fitness_mode=fitness_mode,
        tol_pv=tol_pv,
        tol_delay=tol_delay,
        tol_timescale=tol_timescale,
        tol_resid=tol_resid,
        training_data=training_data,
        mode="1d",
        out_dir=out_dir,
    )

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "ga_fitness.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "fitness"])
        for i, val in enumerate(hists[0]):
            writer.writerow([i, val])

    with open(os.path.join(out_dir, "debug_fitness.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "individual",
                "pv_error",
                "delay_diff",
                "size_penalty",
                "tau_mismatch",
                "pen_species",
                "ignition_shift",
                "reason",
                "keep_count",
                "delta_T",
                "delta_Y",
                "delay_red",
                "fitness",
            ]
        )
        for g, gen in enumerate(debug):
            for i, d in enumerate(gen):
                (
                    pv_err,
                    delay_diff,
                    size_pen,
                    tau_mis,
                    pen_species,
                    ign_shift,
                    reason,
                    keep_cnt,
                    dT,
                    dY,
                    delay_red,
                    fit,
                    genome,
                ) = d
                writer.writerow(
                    [g, i, pv_err, delay_diff, size_pen, tau_mis, pen_species, ign_shift, reason, keep_cnt, dT, dY, delay_red, fit]
                )

    best_sel = sols[0]
    keep = [mech.species_names[i] for i, bit in enumerate(best_sel) if bit]
    red_mech = Mechanism(mech_path)
    red_mech.remove_species([s for s in red_mech.species_names if s not in keep])

    with open(os.path.join(out_dir, "reduced_species.txt"), "w") as f:
        for s in keep:
            f.write(s + "\n")

    red_base = runner(
        red_mech.solution,
        T0_base,
        p0_base,
        Y0,
        tf_total,
        nsteps=len(full.time) - 1,
        use_mole=False,
        time_grid=full.time,
    )

    base_metrics = _compute_case_metrics(
        full,
        red_base,
        mech.species_names,
        red_mech.species_names,
        weights,
        tol_pv,
        tol_delay,
        tol_timescale,
        tol_resid,
    )
    logger.info(
        "Base 1D metrics: pv_err=%.3e delay=%.3e tau_mis=%.3e pen=%.3e shift=%.3e",
        base_metrics["pv_err"],
        base_metrics["delay_metric"],
        base_metrics["tau_mis"],
        base_metrics["pen_species"],
        base_metrics["ign_shift"],
    )

    critical = [
        s
        for s in [
            "CH4",
            "O2",
            "N2",
            "H2O",
            "CO",
            "CO2",
            "OH",
            "H",
            "O",
            "HO2",
            "H2O2",
            "H2",
        ]
        if s in mech.species_names
    ]
    critical_idxs = [mech.species_names.index(s) for s in critical]
    train_details: list[dict] = []
    evaluate_selection_multi(
        sols[0],
        mech,
        weights,
        critical_idxs,
        runner,
        training_data,
        fitness_mode=fitness_mode,
        tol_pv=tol_pv,
        tol_delay=tol_delay,
        tol_timescale=tol_timescale,
        tol_resid=tol_resid,
        target_species=target_species,
        details=train_details,
    )

    wp5_cases: list[dict] = []
    for name in ("POX", "HP_POX", "CO2_recycle"):
        if name in envelopes:
            wp5_cases.extend(_two_point_cases(name, envelopes[name]))
    plasma_cases = _two_point_cases("PLASMA", envelopes["PLASMA"]) if "PLASMA" in envelopes else []

    profiles_rows: list[dict] = []
    kpi_rows: list[dict] = []
    robustness_wp5: list[dict] = []
    robustness_plasma: list[dict] = []
    overlay_cases: list[dict] = []

    def _evaluate_case(case: dict, use_seed: bool) -> None:
        case_id = str(case["id"])
        cached = training_lookup.get(case_id)
        if cached is not None:
            Y_case = dict(cached["Y0"])
            full_case = cached["full"]
        else:
            seed = seed_dict if use_seed and seed_dict else None
            Y_case = _compose_feed(
                mech.solution,
                case["phi"],
                diluent,
                case.get("diluents", []),
                radical_seed=seed,
                co2_ratio=case.get("co2_ch4_ratio"),
            )
            full_case = runner(
                mech.solution,
                case["T0"],
                case["p"],
                Y_case,
                tf_total,
                nsteps=steps_effective,
                use_mole=False,
            )

        red_case = runner(
            red_mech.solution,
            case["T0"],
            case["p"],
            Y_case,
            tf_total,
            nsteps=len(full_case.time) - 1,
            use_mole=False,
            time_grid=full_case.time,
        )

        metrics = _compute_case_metrics(
            full_case,
            red_case,
            mech.species_names,
            red_mech.species_names,
            weights,
            tol_pv,
            tol_delay,
            tol_timescale,
            tol_resid,
        )

        ch4_full = float(full_case.ch4_conversion[-1])
        ch4_red = float(red_case.ch4_conversion[-1])
        co2_full = float(full_case.co2_conversion[-1])
        co2_red = float(red_case.co2_conversion[-1])
        h2co_full = float(full_case.h2_co_ratio[-1])
        h2co_red = float(red_case.h2_co_ratio[-1])
        ign_full = float(full_case.ignition_position)
        ign_red = float(red_case.ignition_position)

        kpi_rows.append(
            {
                "case_id": case["id"],
                "application": case["application"],
                "phi": case["phi"],
                "T0": case["T0"],
                "p": case["p"],
                "CH4_full": ch4_full,
                "CH4_red": ch4_red,
                "CO2_full": co2_full,
                "CO2_red": co2_red,
                "H2CO_full": h2co_full,
                "H2CO_red": h2co_red,
                "ignition_full": ign_full,
                "ignition_red": ign_red,
                "pv_err": metrics["pv_err"],
                "delay_metric": metrics["delay_metric"],
                "tau_mis": metrics["tau_mis"],
                "pen_species": metrics["pen_species"],
                "ign_shift": metrics["ign_shift"],
                "passes": metrics["passes"],
            }
        )

        entry = {
            "case_id": case["id"],
            "application": case["application"],
            "pv_err": metrics["pv_err"],
            "delay_metric": metrics["delay_metric"],
            "tau_mis": metrics["tau_mis"],
            "pen_species": metrics["pen_species"],
            "ign_shift": metrics["ign_shift"],
            "score": metrics["score"],
            "passes": metrics["passes"],
        }
        if case["application"] == "PLASMA":
            robustness_plasma.append(entry)
        else:
            robustness_wp5.append(entry)

        overlay_cases.append(
            {
                "id": case_id,
                "application": case["application"],
                "full": full_case,
                "red": red_case,
                "ign_full": ign_full,
                "ign_red": ign_red,
            }
        )

    for case in wp5_cases:
        _evaluate_case(case, use_seed=False)
    for case in plasma_cases:
        _evaluate_case(case, use_seed=True)

    def _select_top_species(cases: list[dict], n: int) -> list[str]:
        peaks: dict[str, float] = {}
        for data in cases:
            spec_map = data.get("species", {}) or {}
            for name, series in spec_map.items():
                full_vals = np.asarray(series[0], dtype=float)
                if full_vals.size:
                    peaks[name] = max(peaks.get(name, 0.0), float(np.nanmax(np.abs(full_vals))))
        ranked = sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in ranked[:n]]

    plot_cases: list[dict] = []
    for entry in overlay_cases:
        full_case = entry["full"]
        red_case = entry["red"]
        map_full = {s: i for i, s in enumerate(full_case.species_names)}
        map_red = {s: i for i, s in enumerate(red_case.species_names)}
        species_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sp, idx_f in map_full.items():
            yf = full_case.mass_fractions[:, idx_f]
            idx_r = map_red.get(sp)
            yr = red_case.mass_fractions[:, idx_r] if idx_r is not None else np.zeros_like(yf)
            species_map[sp] = (yf, yr)
        entry["species"] = species_map
        entry["x"] = full_case.x
        entry["T_full"] = full_case.temperature
        entry["T_red"] = red_case.temperature
        plot_cases.append(
            {
                "id": entry["id"],
                "application": entry["application"],
                "x": entry["x"],
                "T_full": entry["T_full"],
                "T_red": entry["T_red"],
                "ign_full": entry["ign_full"],
                "ign_red": entry["ign_red"],
                "species": species_map,
            }
        )

    selected_species = _select_top_species(plot_cases, topn_species)
    profiles_rows = []
    for entry in overlay_cases:
        full_case = entry["full"]
        red_case = entry["red"]
        map_full = {s: i for i, s in enumerate(full_case.species_names)}
        map_red = {s: i for i, s in enumerate(red_case.species_names)}
        for j, x_val in enumerate(full_case.x):
            row = {
                "case_id": entry["id"],
                "application": entry["application"],
                "x": float(x_val),
                "temperature_full": float(full_case.temperature[j]),
                "temperature_red": float(red_case.temperature[j]),
                "ch4_conv_full": float(full_case.ch4_conversion[j]),
                "ch4_conv_red": float(red_case.ch4_conversion[j]),
                "h2co_full": float(full_case.h2_co_ratio[j]),
                "h2co_red": float(red_case.h2_co_ratio[j]),
            }
            for sp in selected_species:
                idx_f = map_full.get(sp)
                idx_r = map_red.get(sp)
                row[f"{sp}_full"] = (
                    float(full_case.mass_fractions[j, idx_f]) if idx_f is not None else 0.0
                )
                row[f"{sp}_red"] = (
                    float(red_case.mass_fractions[j, idx_r]) if idx_r is not None else 0.0
                )
            profiles_rows.append(row)

    with open(os.path.join(out_dir, "pfr_profiles.csv"), "w", newline="") as f:
        if profiles_rows:
            fieldnames = list(profiles_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in profiles_rows:
                writer.writerow(row)

    with open(os.path.join(out_dir, "pfr_kpis.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=kpi_rows[0].keys()) if kpi_rows else None
        if writer:
            writer.writeheader()
            for row in kpi_rows:
                writer.writerow(row)

    with open(os.path.join(out_dir, "train_summary.csv"), "w", newline="") as f:
        if train_details:
            fieldnames = list(train_details[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in train_details:
                writer.writerow(row)

    if robustness_wp5:
        with open(os.path.join(out_dir, "robustness_1d.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=robustness_wp5[0].keys())
            writer.writeheader()
            for row in robustness_wp5:
                writer.writerow(row)
    if robustness_plasma:
        with open(os.path.join(out_dir, "robustness_plasma.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=robustness_plasma[0].keys())
            writer.writeheader()
            for row in robustness_plasma:
                writer.writerow(row)

    df_kpi = pd.DataFrame(kpi_rows)
    df_wp5 = pd.DataFrame(robustness_wp5)
    df_plasma = pd.DataFrame(robustness_plasma)

    latex_dir = os.path.join("results", "latex")
    os.makedirs(latex_dir, exist_ok=True)
    if not df_wp5.empty:
        df_wp5.to_latex(os.path.join(latex_dir, "robustness_pox.tex"), index=False, float_format="%.3f")
    if not df_plasma.empty:
        df_plasma.to_latex(os.path.join(latex_dir, "robustness_plasma.tex"), index=False, float_format="%.3f")
    if not df_kpi.empty:
        df_kpi.to_latex(os.path.join(latex_dir, "kpi_summary.tex"), index=False, float_format="%.3f")

    viz_dir = os.path.join(out_dir, "visualizations")
    plot_axial_overlays(plot_cases, None, os.path.join(viz_dir, "axial_overlay"), topn=topn_species)
    plot_kpi_bars(df_kpi, os.path.join(viz_dir, "kpi_bars"))
    plot_consistency_stub(df_kpi, os.path.join(viz_dir, "consistency"), baseline=None)

    logger.info("1D pipeline completed with %d WP5 cases and %d plasma cases", len(wp5_cases), len(plasma_cases))


def full_pipeline(
    mech_path: str,
    out_dir: str,
    steps: int = 1000,
    tf: float = 0.5,
    phi: float | None = None,
    preset: str = "methane_air",
    T0: float | None = None,
    p0: float | None = None,
    log_times: bool = False,
    alpha: float = 0.8,
    tf_short: float | None = None,
    steps_short: int | None = None,
    cache_labels: bool = True,
    isothermal: bool = False,
    min_species: int | None = None,
    max_species: int | None = None,
    target_species: int | None = None,
    generations: int = 60,
    population: int = 40,
    mutation: float = 0.25,
    focus: str = "auto",
    focus_window: Tuple[float, float] | None = None,
    report_grid: str | None = None,
    *,
    mode: str = "0d",
    diluent: str = "N2",
    L: float = 0.5,
    D: float = 0.05,
    mdot: float = 0.1,
    U: float = 0.0,
    Tw: float | None = None,
    fitness_mode: str = "standard",
    tol_pv: float = 0.05,
    tol_delay: float = 0.05,
    tol_timescale: float = 0.05,
    tol_resid: float = 0.05,
    plasma_length: float = 0.0,
    T_plasma_out: float | None = None,
    radical_seed: Dict[str, float] | None = None,
    envelopes_path: str = "envelopes.json",
    train_cases: str = "auto",
    train_weights: str | None = None,
    topn_species: int = 6,
) -> None:
    if mode == "1d":
        _full_pipeline_pfr(
            mech_path,
            out_dir,
            steps=steps,
            phi=phi,
            T0=T0,
            p0=p0,
            log_times=log_times,
            alpha=alpha,
            tf_short=tf_short,
            steps_short=steps_short,
            cache_labels=cache_labels,
            min_species=min_species,
            max_species=max_species,
            target_species=target_species,
            generations=generations,
            population=population,
            mutation=mutation,
        fitness_mode=fitness_mode,
        tol_pv=tol_pv,
        tol_delay=tol_delay,
        tol_timescale=tol_timescale,
        tol_resid=tol_resid,
        diluent=diluent,
        L=L,
        D=D,
        mdot=mdot,
        U=U,
        Tw=Tw,
        plasma_length=plasma_length,
        T_plasma_out=T_plasma_out,
        radical_seed=radical_seed,
        envelopes_path=envelopes_path,
        train_cases=train_cases,
        train_weights=train_weights,
        topn_species=topn_species,
    )
    else:
        _full_pipeline_batch(
            mech_path,
            out_dir,
            steps=steps,
            tf=tf,
            phi=phi,
            preset=preset,
            T0=T0,
            p0=p0,
            log_times=log_times,
            alpha=alpha,
            tf_short=tf_short,
            steps_short=steps_short,
            cache_labels=cache_labels,
            isothermal=isothermal,
            min_species=min_species,
            max_species=max_species,
            target_species=target_species,
            generations=generations,
            population=population,
            mutation=mutation,
            focus=focus,
            focus_window=focus_window,
            report_grid=report_grid,
        )

