"""GA-GNN reduction pipeline targeting HP-POX KPIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import json
import time

import numpy as np
import pandas as pd

import cantera as ct

from metaheuristics.ga import GAOptions, run_ga

from .configuration import CaseDefinition, load_case_definition
from .pfr import PlugFlowOptions, PlugFlowSolver


@dataclass
class ReductionConfig:
    mechanism: str
    cases: Sequence[str] = ("Case1", "Case4")
    population_size: int = 30
    generations: int = 20
    min_species: int = 20
    max_species: int | None = 80
    gnn_seed_path: str | None = None
    output_dir: Path = Path("results/hp_pox_reduction")
    metric_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "CH4_conversion": 1.0,
            "CO2_conversion": 0.5,
            "H2_CO_ratio": 1.5,
            "pressure_drop_Pa": 0.1,
            "ignition_position_m": 1.0,
        }
    )


@dataclass
class ReductionResult:
    best_genome: np.ndarray
    history: List[float]
    debug: List[List[tuple]]
    baseline_metrics: Dict[str, Mapping[str, float]]
    best_metrics: Dict[str, Mapping[str, float]]
    species_names: Sequence[str]
    output_dir: Path


class GAGNNReducer:
    def __init__(self, config: ReductionConfig) -> None:
        self.config = config
        self.mechanism = ct.Solution(config.mechanism)
        self.species_names = self.mechanism.species_names
        self.reference_cases = [load_case_definition(case) for case in config.cases]
        self.baseline_metrics = self._compute_baseline_metrics()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_species = self._determine_fixed_species()
        self.seeds = self._load_seed_population()

    # ------------------------------------------------------------------
    def run(self) -> ReductionResult:
        genome_length = len(self.species_names)
        ga_options = GAOptions(
            population_size=self.config.population_size,
            generations=self.config.generations,
            min_species=self.config.min_species,
            max_species=self.config.max_species,
        )
        start = time.perf_counter()
        best, history, debug = run_ga(
            genome_length=genome_length,
            eval_fn=self._evaluate_individual,
            options=ga_options,
            return_history=True,
            return_debug=True,
            initial_population=self.seeds,
            fixed_indices=[self.species_names.index(name) for name in self.fixed_species],
        )
        elapsed = time.perf_counter() - start
        best_metrics = self._metrics_for_genome(best)
        self._export_results(best, history, debug, best_metrics, elapsed)
        return ReductionResult(
            best_genome=best,
            history=history,
            debug=debug,
            baseline_metrics=self.baseline_metrics,
            best_metrics=best_metrics,
            species_names=self.species_names,
            output_dir=self.output_dir,
        )

    # ------------------------------------------------------------------
    def _evaluate_individual(self, genome: np.ndarray) -> tuple[float, Dict[str, float]]:
        metrics = self._metrics_for_genome(genome)
        if not metrics:
            return -1e6, {}
        error = 0.0
        for case_name, case_metrics in self.baseline_metrics.items():
            cand = metrics.get(case_name, {})
            for metric_name, weight in self.config.metric_weights.items():
                ref_value = case_metrics.get(metric_name)
                cand_value = cand.get(metric_name)
                if ref_value is None or cand_value is None:
                    continue
                if ref_value == 0:
                    diff = abs(cand_value - ref_value)
                else:
                    diff = abs((cand_value - ref_value) / ref_value)
                error += weight * diff
        score = -error
        return score, metrics

    def _metrics_for_genome(self, genome: np.ndarray) -> Dict[str, Dict[str, float]]:
        active_species = [name for i, name in enumerate(self.species_names) if genome[i] > 0.5]
        reduced_gas = self._build_reduced_solution(active_species)
        metrics: Dict[str, Dict[str, float]] = {}
        for case in self.reference_cases:
            try:
                solver = PlugFlowSolver(reduced_gas, case, PlugFlowOptions(output_points=120))
                result = solver.solve()
            except Exception:
                return {}
            metrics[case.name] = result.metrics
        if len(metrics) != len(self.reference_cases):
            return {}
        return metrics

    def _build_reduced_solution(self, active_species: Sequence[str]) -> ct.Solution:
        species = [sp for sp in self.mechanism.species() if sp.name in active_species]
        allowed = {sp.name for sp in species}
        reactions = [
            rxn
            for rxn in self.mechanism.reactions()
            if set(rxn.reactants).issubset(allowed) and set(rxn.products).issubset(allowed)
        ]
        if not species:
            raise ValueError("No species selected for reduced mechanism")
        return ct.Solution(thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=reactions)

    def _compute_baseline_metrics(self) -> Dict[str, Mapping[str, float]]:
        metrics: Dict[str, Mapping[str, float]] = {}
        for case in self.reference_cases:
            solver = PlugFlowSolver(self.mechanism, case, PlugFlowOptions(output_points=120))
            result = solver.solve()
            metrics[case.name] = result.metrics
        return metrics

    def _determine_fixed_species(self) -> Sequence[str]:
        essential = {"CH4", "H2", "CO", "CO2", "H2O", "O2", "N2"}
        for case in self.reference_cases:
            for stream in case.streams:
                essential.update(stream.composition.keys())
        return [name for name in essential if name in self.species_names]

    def _load_seed_population(self) -> np.ndarray:
        if self.config.gnn_seed_path is None:
            return None
        path = Path(self.config.gnn_seed_path)
        if not path.exists():
            return None
        scores = pd.read_csv(path)
        species_to_idx = {name: i for i, name in enumerate(self.species_names)}
        ranked = sorted(
            ((species_to_idx[row["species"]], row["score"]) for _, row in scores.iterrows() if row["species"] in species_to_idx),
            key=lambda item: item[1],
            reverse=True,
        )
        genome_length = len(self.species_names)
        seeds: List[np.ndarray] = []
        top_indices = [idx for idx, _ in ranked[: self.config.max_species or 80]]
        seed = np.zeros(genome_length, dtype=int)
        seed[top_indices] = 1
        seeds.append(seed)
        return np.array(seeds)

    def _export_results(
        self,
        best: np.ndarray,
        history: Sequence[float],
        debug: Sequence[Sequence[tuple]],
        best_metrics: Mapping[str, Mapping[str, float]],
        elapsed: float,
    ) -> None:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "best_genome.npy", best)
        pd.DataFrame({"generation": range(len(history)), "score": history}).to_csv(
            out_dir / "fitness_history.csv", index=False
        )
        with (out_dir / "baseline_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(self.baseline_metrics, handle, indent=2)
        with (out_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metrics, handle, indent=2)
        summary = {
            "runtime_s": elapsed,
            "baseline_species": len(self.species_names),
            "best_species": int(best.sum()),
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
