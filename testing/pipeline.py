"""Utilities for running the full reduction pipeline and plotting results."""

import csv
import json
import os
from typing import List, Tuple, Sequence
import numpy as np
import cantera as ct
from mechanism.loader import Mechanism
from reactor.batch import BatchResult, run_constant_pressure
from progress_variable import progress_variable
from metaheuristics.ga import run_ga, GAOptions
from metrics import pv_error, ignition_delay
from graph.construction import build_species_graph
from gnn.models import train_dummy_gnn, predict_scores
from visualizations import (
    plot_mole_fraction,
    plot_ignition_delays,
    plot_convergence,
    plot_pv_errors,
)


def evaluate_selection(
    selection: np.ndarray,
    base_mech: Mechanism,
    Y0: dict,
    tf: float,
    full_res,
    weights: Sequence[float] | None = None,
    critical_idxs: Sequence[int] | None = None,
) -> Tuple[float, float, float, str]:
    """Return fitness and debug info for a given species selection.

    Parameters
    ----------
    selection:
        Binary vector indicating which species to keep.
    base_mech:
        Reference mechanism used as a template.
    Y0, tf:
        Initial composition and final time for the reactor simulation.
    full_res:
        Result from the full mechanism simulation used for comparison.
    weights:
        Progress-variable weights.
    critical_idxs:
        Indices of species that must always be retained.
    """

    size_frac = float(selection.sum()) / len(selection)
    reason = ""
    if critical_idxs is not None:
        if selection.sum() < max(4, len(critical_idxs)) or np.any(selection[list(critical_idxs)] == 0):
            return -1e6, 1.0, size_frac, "missing_critical"

    idx_keep = [i for i, bit in enumerate(selection) if bit]
    keep = [base_mech.species_names[i] for i in idx_keep]
    mech = Mechanism(base_mech.file_path)
    remove = [s for s in mech.species_names if s not in keep]
    print("Species to remove:", remove)
    print(f"Species retained: {selection.sum()}, Total: {len(selection)}")
    try:
        if remove:
            mech.remove_species(remove)
        res = run_constant_pressure(
            mech.solution,
            full_res.temperature[0],
            ct.one_atm,
            Y0,
            tf,
            nsteps=len(full_res.time),
        )
    except Exception as e:
        return -1e6, 1.0, size_frac, f"sim_failed:{type(e).__name__}"
    if weights is None:
        weights_sel = np.ones(len(idx_keep))
    else:
        weights_sel = np.array([weights[i] for i in idx_keep])
    idx_red = [mech.species_names.index(s) for s in keep]
    pv_full = progress_variable(full_res.mass_fractions[:, idx_keep], weights_sel)
    pv_red = progress_variable(res.mass_fractions[:, idx_red], weights_sel)
    err = pv_error(pv_full, pv_red)
    fitness = -err - 0.1 * size_frac
    return fitness, err, size_frac, reason


def run_ga_reduction(
    mech_path: str,
    Y0: dict,
    tf: float,
    steps: int = 200,
) -> Tuple[
    List[str],
    List[np.ndarray],
    List[List[float]],
    List[List[tuple]],
    BatchResult,
    np.ndarray,
]:
    """Run the GA-based reduction on the given mechanism."""

    mech = Mechanism(mech_path)
    full = run_constant_pressure(mech.solution, 1000.0, ct.one_atm, Y0, tf, nsteps=steps)
    genome_len = len(mech.species_names)

    weights_path = os.path.join("data", "species_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            weight_map = json.load(f)
        weights = np.array([weight_map.get(s, 1e-3) for s in mech.species_names])
    else:
        weights = np.ones(genome_len)

    G = build_species_graph(mech.solution)
    gnn_model = train_dummy_gnn(G, epochs=5)
    scores_dict = predict_scores(gnn_model, G)
    scores = np.array([scores_dict[s] for s in mech.species_names])

    critical = [s for s in ["CH4", "O2", "N2"] if s in mech.species_names]
    critical_idxs = [mech.species_names.index(s) for s in critical]

    order = np.argsort(scores)
    k = int(0.2 * genome_len)
    pop_size = 12
    num_seed = max(1, pop_size // 2)
    init_pop = np.zeros((pop_size, genome_len), dtype=int)

    # first seed keeps only top-k species by score
    seed_topk = np.zeros(genome_len, dtype=int)
    seed_topk[order[-k:]] = 1
    seed_topk[critical_idxs] = 1
    init_pop[0] = seed_topk

    # subsequent seeds start from all ones and remove lowest scoring species
    base = np.ones(genome_len, dtype=int)
    base[critical_idxs] = 1
    for i in range(1, num_seed):
        seed = base.copy()
        removed = 0
        for idx in order:
            if idx not in critical_idxs:
                seed[idx] = 0
                removed += 1
                if removed == i:
                    break
        init_pop[i] = seed

    for i in range(num_seed, pop_size):
        init_pop[i] = np.random.randint(0, 2, genome_len)
        init_pop[i][critical_idxs] = 1

    eval_fn = lambda sel: evaluate_selection(
        sel, mech, Y0, tf, full, weights, critical_idxs
    )

    ga_sel, ga_hist, debug = run_ga(
        genome_len,
        eval_fn,
        GAOptions(population_size=pop_size, generations=6),
        return_history=True,
        initial_population=init_pop,
        return_debug=True,
        fixed_indices=critical_idxs,
    )
    print(f"Best selection retains {int(ga_sel.sum())} of {genome_len} species")
    with open("best_selection.txt", "w") as f:
        f.write(",".join(map(str, ga_sel.tolist())))

    names = ["GA"]
    solutions = [ga_sel]
    history = [ga_hist]
    return names, solutions, history, debug, full, weights


def save_metrics(
    names: List[str],
    sols: List[np.ndarray],
    full_res: BatchResult,
    mech: Mechanism,
    Y0: dict,
    tf: float,
    out_dir: str,
    weights: Sequence[float] | None = None,
) -> Tuple[List[BatchResult], List[float], List[float], List[int]]:
    """Evaluate reduced mechanisms and store metric CSV files."""

    os.makedirs(out_dir, exist_ok=True)
    results: List[BatchResult] = []
    pv_errors: List[float] = []
    delays: List[float] = []
    sizes: List[int] = []
    if weights is None:
        weights = np.ones(full_res.mass_fractions.shape[1])
    full_delay = ignition_delay(full_res.time, full_res.temperature)

    metrics_rows = []
    for name, sel in zip(names, sols):
        idx_keep = [i for i, b in enumerate(sel) if b]
        keep = [mech.species_names[i] for i in idx_keep]
        red_mech = Mechanism(mech.file_path)
        remove = [s for s in red_mech.species_names if s not in keep]
        try:
            if remove:
                red_mech.remove_species(remove)
            res = run_constant_pressure(
                red_mech.solution,
                1000.0,
                ct.one_atm,
                Y0,
                tf,
                nsteps=len(full_res.time),
            )
            weights_sel = np.array([weights[i] for i in idx_keep])
            idx_red = [red_mech.species_names.index(s) for s in keep]
            pv_full = progress_variable(full_res.mass_fractions[:, idx_keep], weights_sel)
            pv_red = progress_variable(res.mass_fractions[:, idx_red], weights_sel)
            err = pv_error(pv_full, pv_red)
            delay = ignition_delay(res.time, res.temperature)
            size_red = int(sel.sum())
            metrics_rows.append([
                name,
                err,
                abs(delay - full_delay) / full_delay,
                size_red,
            ])
            results.append(res)
            pv_errors.append(err)
            delays.append(delay)
            sizes.append(size_red)
        except Exception:
            metrics_rows.append([name, 1.0, 1.0, 0])
            results.append(full_res)
            pv_errors.append(1.0)
            delays.append(full_delay)
            sizes.append(0)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm",
            "pv_error",
            "ignition_delay_error",
            "species_retained",
        ])
        writer.writerows(metrics_rows)

    with open(os.path.join(out_dir, "ignition_delay.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "ignition_delay_s"])
        writer.writerow(["Full", full_delay])
        writer.writerows(zip(names, delays))

    with open(os.path.join(out_dir, "pv_error.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "pv_error"])
        writer.writerows(zip(names, pv_errors))

    with open(os.path.join(out_dir, "species_retained.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "num_species"])
        writer.writerows(zip(names, sizes))

    return results, pv_errors, delays, sizes


import matplotlib.pyplot as plt

def plot_profiles(
    full_res: BatchResult,
    red_res: BatchResult,
    species_idx: Sequence[int],
    species_names: Sequence[str],
    out_base: str,
) -> None:
    """Plot mole-fraction profiles of selected species."""

    fig, ax = plt.subplots()
    for idx, name in zip(species_idx, species_names):
        ax.plot(full_res.time, full_res.mass_fractions[:, idx], label=f"{name} full")
        ax.plot(red_res.time, red_res.mass_fractions[:, idx], "--", label=f"{name} red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass Fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".pdf")
    plt.close(fig)

def full_pipeline(mech_path: str, out_dir: str, steps: int = 200, tf: float = 1.0):
    """Run the complete testing pipeline and generate plots/CSVs."""

    Y0 = {"CH4": 0.5, "O2": 1.0, "N2": 3.76}

    names, solutions, histories, debug, full, weights = run_ga_reduction(
        mech_path, Y0, tf, steps
    )
    mech = Mechanism(mech_path)

    results, pv_err, delays, sizes = save_metrics(
        names, solutions, full, mech, Y0, tf, out_dir, weights
    )

    # write fitness history CSVs
    for name, hist in zip(names, histories):
        if not hist:
            continue
        path = os.path.join(out_dir, f"{name.lower()}_fitness.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "fitness"])
            for i, val in enumerate(hist):
                writer.writerow([i, val])

    # debug fitness logging
    dbg_path = os.path.join(out_dir, "debug_fitness.csv")
    with open(dbg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "individual", "pv_error", "size_penalty", "fitness", "reason"])
        for g, gen in enumerate(debug):
            for i, det in enumerate(gen):
                pv_err_g, size_pen, fit, reason, genome = det
                writer.writerow([g, i, pv_err_g, size_pen, fit, reason])

    # visual debugging for best individual of each generation
    for g, gen in enumerate(debug):
        if not gen:
            continue
        scores = [d[2] for d in gen]
        best_idx = int(np.argmax(scores))
        genome = gen[best_idx][4]
        idx_keep = [i for i, bit in enumerate(genome) if bit]
        keep = [mech.species_names[i] for i in idx_keep]
        red_mech = Mechanism(mech_path)
        remove = [s for s in red_mech.species_names if s not in keep]
        try:
            if remove:
                red_mech.remove_species(remove)
            res = run_constant_pressure(
                red_mech.solution,
                1000.0,
                ct.one_atm,
                Y0,
                tf,
                nsteps=len(full.time),
            )
            weights_sel = np.array([weights[i] for i in idx_keep])
            pv_full = progress_variable(full.mass_fractions[:, idx_keep], weights_sel)
            pv_red = progress_variable(res.mass_fractions, weights_sel)

            fig, ax = plt.subplots()
            ax.plot(full.time, pv_full, label="full")
            ax.plot(res.time, pv_red, "--", label="reduced")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Progress Variable")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"debug_gen{g}_pv.png"))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(full.time, full.temperature, label="full")
            ax.plot(res.time, res.temperature, "--", label="reduced")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Temperature [K]")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"debug_gen{g}_T.png"))
            plt.close(fig)
        except Exception:
            continue

    # choose the algorithm with minimum PV error for profile plot
    best_idx = int(np.argmin(pv_err)) if pv_err else 0
    key_species = [s for s in ["CH4", "O2", "CO2"] if s in mech.species_names]
    idxs = [mech.species_names.index(s) for s in key_species]
    if results:
        plot_profiles(
            full,
            results[best_idx],
            idxs,
            key_species,
            os.path.join(out_dir, "profiles"),
        )

    plot_ignition_delays(delays, names, os.path.join(out_dir, "ignition_delay"))
    plot_convergence(histories, names, os.path.join(out_dir, "convergence"))
    plot_pv_errors(pv_err, names, os.path.join(out_dir, "pv_error"))