import csv
import os
from typing import Callable, List, Tuple, Sequence
import numpy as np
import cantera as ct
from mechanism.loader import Mechanism
from reactor.batch import BatchResult, run_constant_pressure
from progress_variable import progress_variable, optimise_weights
from metaheuristics import (
    run_ga, GAOptions,
    run_abc, ABCOptions,
    run_bees, BeesOptions,
    run_bnb, BnBOptions,
)
from visualizations import plot_convergence, plot_comparison
from graph.construction import build_species_graph
from gnn.models import train_dummy_gnn, predict_scores


def pv_error(full: np.ndarray, red: np.ndarray) -> float:
    return np.linalg.norm(full - red) / (np.linalg.norm(full) + 1e-12)


def ignition_delay(time: np.ndarray, T: np.ndarray, Y: np.ndarray | None = None, species_idx: int | None = None) -> float:
    """Return ignition delay based on temperature or species derivative.

    If ``species_idx`` is provided, the ignition delay is taken from the peak of
    ``dY[:, species_idx]/dt``; otherwise the peak of ``dT/dt`` is used.
    """

    if species_idx is not None and Y is not None:
        deriv = np.gradient(Y[:, species_idx], time)
    else:
        deriv = np.gradient(T, time)
    idx = int(np.argmax(deriv))
    return float(time[idx])


def evaluate_selection(
    selection: np.ndarray,
    base_mech: Mechanism,
    Y0: dict,
    tf: float,
    full_res,
    scores: np.ndarray | None = None,
    weights: Sequence[float] | None = None,
) -> float:
    """Return fitness for a given species selection."""

    keep = [base_mech.species_names[i] for i, bit in enumerate(selection) if bit]
    mech = Mechanism(base_mech.file_path)
    remove = [s for s in mech.species_names if s not in keep]
    try:
        if remove:
            mech.remove_species(remove)
        res = run_constant_pressure(mech.solution, full_res.temperature[0], ct.one_atm, Y0, tf, nsteps=len(full_res.time))
    except Exception:
        return -1e6
    if weights is None:
        weights = np.ones(full_res.mass_fractions.shape[1])
    pv_full = progress_variable(full_res.mass_fractions, weights)
    pv_red = progress_variable(res.mass_fractions, weights)
    err = pv_error(pv_full, pv_red)
    if scores is None:
        size_penalty = selection.sum() / len(selection)
    else:
        size_penalty = (1 - selection) * scores
        size_penalty = size_penalty.sum() / len(selection)
    return -(err + 0.01 * float(size_penalty))


def run_all_algorithms(
    mech_path: str,
    Y0: dict,
    tf: float,
    steps: int = 200,
) -> Tuple[List[str], List[np.ndarray], List[List[float]], List[float], BatchResult, np.ndarray]:
    """Run all metaheuristics on the given mechanism."""

    mech = Mechanism(mech_path)
    full = run_constant_pressure(mech.solution, 1000.0, ct.one_atm, Y0, tf, nsteps=steps)
    genome_len = len(mech.species_names)

    # optimise PV weights from the full solution
    pv_ref = progress_variable(full.mass_fractions, np.ones(genome_len))
    weights = optimise_weights(full.mass_fractions, pv_ref)

    G = build_species_graph(mech.solution)
    gnn_model = train_dummy_gnn(G, epochs=5)
    scores_dict = predict_scores(gnn_model, G)
    scores = np.array([scores_dict[s] for s in mech.species_names])

    eval_fn = lambda sel: evaluate_selection(sel, mech, Y0, tf, full, scores, weights)

    import time

    runtimes: List[float] = []

    t0 = time.perf_counter()
    ga_sel, ga_hist = run_ga(genome_len, eval_fn, GAOptions(generations=5))
    runtimes.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    abc_sel, abc_hist = run_abc(genome_len, eval_fn)
    runtimes.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    bees_sel, bees_hist = run_bees(genome_len, eval_fn)
    runtimes.append(time.perf_counter() - t0)

    if genome_len <= 6:
        t0 = time.perf_counter()
        bnb_sel, bnb_hist = run_bnb(min(genome_len, 6), eval_fn, BnBOptions(max_size=6))
        runtimes.append(time.perf_counter() - t0)
    else:
        bnb_sel, bnb_hist = np.ones(genome_len, dtype=int), []
        runtimes.append(0.0)

    names = ['GA', 'ABC', 'Bees', 'BnB']
    solutions = [ga_sel, abc_sel, bees_sel, bnb_sel]
    history = [ga_hist, abc_hist, bees_hist, bnb_hist]
    return names, solutions, history, runtimes, full, weights


def save_metrics(
    names: List[str],
    sols: List[np.ndarray],
    runtimes: Sequence[float],
    full_res: BatchResult,
    mech: Mechanism,
    Y0: dict,
    tf: float,
    out_dir: str,
    weights: Sequence[float] | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics = []
    if weights is None:
        weights = np.ones(full_res.mass_fractions.shape[1])
    pv_full = progress_variable(full_res.mass_fractions, weights)
    full_delay = ignition_delay(full_res.time, full_res.temperature)
    for name, sel, rt in zip(names, sols, runtimes):
        keep = [mech.species_names[i] for i, b in enumerate(sel) if b]
        red_mech = Mechanism(mech.file_path)
        remove = [s for s in red_mech.species_names if s not in keep]
        try:
            if remove:
                red_mech.remove_species(remove)
            res = run_constant_pressure(red_mech.solution, 1000.0, ct.one_atm, Y0, tf, nsteps=len(full_res.time))
            pv_red = progress_variable(res.mass_fractions, weights)
            err = pv_error(pv_full, pv_red)
            delay = ignition_delay(res.time, res.temperature)
            size_red = 1 - sel.sum() / len(sel)
            metrics.append([name, err, abs(delay - full_delay)/full_delay, size_red, rt])
        except Exception:
            metrics.append([name, 1.0, 1.0, 0.0, rt])
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'pv_error', 'ignition_delay_error', 'size_reduction', 'runtime'])
        writer.writerows(metrics)


import matplotlib.pyplot as plt

def plot_profiles(full_res, mech: Mechanism, Y0: dict, tf: float, out_dir: str):
    key_species = ['CH4', 'O2', 'CO2']
    plt.figure()
    for sp in key_species:
        if sp in mech.species_names:
            idx = mech.species_names.index(sp)
            plt.plot(full_res.time, full_res.mass_fractions[:, idx], label=sp)
    plt.xlabel('Time [s]')
    plt.ylabel('Mass Fraction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'profiles.png'))
    plt.close()

def full_pipeline(mech_path: str, out_dir: str, steps: int = 200):
    Y0 = {'CH4':0.5, 'O2':1.0, 'N2':3.76}
    tf = 0.02
    names, solutions, history, runtimes, full, weights = run_all_algorithms(mech_path, Y0, tf, steps)
    mech = Mechanism(mech_path)
    save_metrics(names, solutions, runtimes, full, mech, Y0, tf, out_dir, weights)
    plot_profiles(full, mech, Y0, tf, out_dir)
    plot_convergence(history, names, out_dir)
    plot_comparison(os.path.join(out_dir, 'metrics.csv'), out_dir)
