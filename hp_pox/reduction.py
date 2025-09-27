"""GA-GNN reduction pipeline targeting HP-POX KPIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import json
import time

import numpy as np
import pandas as pd

import cantera as ct

from metaheuristics.ga import GAOptions, run_ga

from .configuration import load_case_definition
from .pfr import PlugFlowOptions, PlugFlowResult, PlugFlowSolver
import logging


@dataclass
class ReductionConfig:
    mechanism: str
    cases_1d: Sequence[str] = ("Case1", "Case4")
    cases_0d: Sequence[str] = ()
    population_size: int = 60
    generations: int = 60
    min_species: int = 25
    max_species: int | None = 80
    gnn_seed_path: str | None = None
    keep_core: bool = True
    elitism: int = 1
    seed: int | None = None
    output_dir: Path = Path("results/hp_pox_reduction")
    sample_points: int = 240
    profile_weights: Mapping[str, float] = field(
        default_factory=lambda: {"pre": 1.0, "ignition": 2.0, "post": 1.0}
    )
    species_targets: Sequence[str] = ("CH4", "CO", "H2", "CO2", "H2O")


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
        self.reference_cases = [load_case_definition(case) for case in config.cases_1d]
        self.reference_profiles = self._compute_reference_profiles()
        self.baseline_metrics = self._collect_baseline_metrics()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.required_species = {
            "H",
            "O",
            "OH",
            "HO2",
            "H2O2",
            "CH4",
            "CO",
            "CO2",
            "H2",
            "H2O",
            "CH3",
            "HCO",
        }
        self.fixed_species = self._determine_fixed_species()
        self.required_reaction_patterns = {
            "CO + H2O": set(),
            "CO + OH": set(),
            "CO + O2": set(),
        }
        for reaction in self.mechanism.reactions():
            eq = reaction.equation
            for pattern in self.required_reaction_patterns:
                if pattern in eq:
                    self.required_reaction_patterns[pattern].add(eq)
        self.seeds = self._load_seed_population()
        self.logger = logging.getLogger(__name__)
        if config.seed is not None:
            np.random.seed(config.seed)
        if config.cases_0d:
            self.logger.warning(
                "0-D training cases specified but not yet implemented: %s",
                ", ".join(config.cases_0d),
            )

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
        fixed_indices = [self.species_names.index(name) for name in self.fixed_species]
        best, history, debug = run_ga(
            genome_length=genome_length,
            eval_fn=self._evaluate_individual,
            options=ga_options,
            return_history=True,
            return_debug=True,
            initial_population=self.seeds,
            fixed_indices=fixed_indices,
        )
        elapsed = time.perf_counter() - start
        best_score, best_details = self._evaluate_individual(best)
        best_metrics = {
            case: data["metrics"] for case, data in best_details.get("cases", {}).items()
        }
        self._export_results(best, history, debug, best_details, elapsed)
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
    def _evaluate_individual(self, genome: np.ndarray) -> tuple[float, Dict[str, object]]:
        try:
            details = self._solve_for_genome(genome)
        except Exception as exc:  # noqa: BLE001
            return -1e6, {"error": str(exc)}
        score = -details.get("total_error", 1e6)
        return score, details

    def _solve_for_genome(self, genome: np.ndarray) -> Dict[str, object]:
        active_species = [name for i, name in enumerate(self.species_names) if genome[i] > 0.5]
        missing = self.required_species.difference(active_species)
        if missing:
            raise ValueError(f"Genome removed required species: {sorted(missing)}")
        allowed = set(active_species)
        reactions_data = [
            rxn.input_data
            for rxn in self.mechanism.reactions()
            if set(rxn.reactants).issubset(allowed) and set(rxn.products).issubset(allowed)
        ]
        if not reactions_data:
            raise ValueError("Genome eliminated all reactions")
        penalty = self._reaction_penalty(reactions_data)
        options = PlugFlowOptions(output_points=self.config.sample_points)
        case_results: Dict[str, Dict[str, object]] = {}
        total_error = penalty
        for case in self.reference_cases:
            species_objects = [
                ct.Species.from_dict(self.mechanism.species(name).input_data)
                for name in active_species
            ]
            reduced_gas = ct.Solution(
                thermo="IdealGas",
                kinetics="GasKinetics",
                species=species_objects,
                reactions=[ct.Reaction.from_dict(data) for data in reactions_data],
            )
            solver = PlugFlowSolver(
                reduced_gas,
                case,
                options=options,
            )
            result = solver.solve()
            errors = self._compare_profiles(case.name, result)
            total_error += errors["total"]
            case_results[case.name] = {
                "metrics": result.metrics,
                "errors": errors,
                "species": active_species,
            }
        return {
            "cases": case_results,
            "penalty": penalty,
            "total_error": total_error,
            "active_species": active_species,
            "reactions": len(reactions_data),
        }

    def _compare_profiles(self, case_name: str, candidate: PlugFlowResult) -> Dict[str, object]:
        baseline = self.reference_profiles[case_name]
        xs = baseline.positions_m
        masks = self._region_masks(xs, baseline.metrics.get("ignition_position_m"))
        weights = self.config.profile_weights
        temp_errors: Dict[str, float] = {}
        species_errors: Dict[str, Dict[str, float]] = {}
        total = 0.0
        for region, mask in masks.items():
            if not np.any(mask):
                continue
            temp_err = _relative_l2(
                baseline.temperature_K[mask], candidate.temperature_K[mask]
            )
            temp_errors[region] = temp_err
            total += weights.get(region, 1.0) * temp_err
            for species in self.config.species_targets:
                baseline_profile = baseline.mole_fractions.get(species)
                candidate_profile = candidate.mole_fractions.get(species)
                if baseline_profile is None or candidate_profile is None:
                    continue
                err = _relative_l2(baseline_profile[mask], candidate_profile[mask])
                species_errors.setdefault(species, {})[region] = err
                total += weights.get(region, 1.0) * err
        baseline_ign = baseline.metrics.get("ignition_position_m")
        candidate_ign = candidate.metrics.get("ignition_position_m")
        ignition_penalty = 0.0
        if baseline_ign is not None:
            if candidate_ign is None:
                ignition_penalty = 1.0
            else:
                ignition_penalty = abs(candidate_ign - baseline_ign) / max(baseline_ign, 1e-6)
        total += ignition_penalty
        return {
            "temperature": temp_errors,
            "species": species_errors,
            "ignition_penalty": ignition_penalty,
            "total": total,
        }

    def _region_masks(self, xs: np.ndarray, ignition: float | None) -> Dict[str, np.ndarray]:
        length = xs[-1] - xs[0]
        step = length / max(len(xs) - 1, 1)
        window = max(0.02 * length, step)
        if ignition is None:
            idx1 = max(int(0.3 * len(xs)), 1)
            idx2 = max(int(0.6 * len(xs)), idx1 + 1)
            pre = np.zeros_like(xs, dtype=bool)
            ign = np.zeros_like(xs, dtype=bool)
            post = np.zeros_like(xs, dtype=bool)
            pre[:idx1] = True
            ign[idx1:idx2] = True
            post[idx2:] = True
        else:
            pre = xs <= max(ignition - window, xs[0])
            ign = (xs >= ignition - window) & (xs <= ignition + window)
            post = xs >= min(ignition + window, xs[-1])
            if not np.any(pre):
                pre[np.argmin(np.abs(xs - max(ignition - window, xs[0])))] = True
            if not np.any(ign):
                ign[np.argmin(np.abs(xs - ignition))] = True
            if not np.any(post):
                post[-1] = True
        return {"pre": pre, "ignition": ign, "post": post}

    def _reaction_penalty(self, reactions: Sequence[Mapping[str, object]]) -> float:
        equations = {reaction["equation"] for reaction in reactions if "equation" in reaction}
        missing = [
            pattern
            for pattern, matches in self.required_reaction_patterns.items()
            if matches and not (equations & matches)
        ]
        return float(len(missing)) * 10.0

    def _compute_reference_profiles(self) -> Dict[str, PlugFlowResult]:
        profiles: Dict[str, PlugFlowResult] = {}
        options = PlugFlowOptions(output_points=self.config.sample_points)
        for case in self.reference_cases:
            solver = PlugFlowSolver(self.mechanism, case, options)
            profiles[case.name] = solver.solve()
        return profiles

    def _collect_baseline_metrics(self) -> Dict[str, Mapping[str, float]]:
        return {name: result.metrics for name, result in self.reference_profiles.items()}

    def _determine_fixed_species(self) -> Sequence[str]:
        essential = set(self.required_species)
        if self.config.keep_core:
            essential.update({"O2", "N2"})
        for case in self.reference_cases:
            for stream in case.streams:
                essential.update(stream.composition.keys())
        return [name for name in essential if name in self.species_names]

    def _solution_from_genome(self, genome: np.ndarray) -> ct.Solution:
        active_species = [self.species_names[i] for i in range(len(self.species_names)) if genome[i] > 0.5]
        allowed = set(active_species)
        species_objects = [
            ct.Species.from_dict(self.mechanism.species(name).input_data)
            for name in active_species
        ]
        reactions = [
            ct.Reaction.from_dict(rxn.input_data)
            for rxn in self.mechanism.reactions()
            if set(rxn.reactants).issubset(allowed) and set(rxn.products).issubset(allowed)
        ]
        return ct.Solution(
            thermo="IdealGas",
            kinetics="GasKinetics",
            species=species_objects,
            reactions=reactions,
        )

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
        for name in self.fixed_species:
            seed[self.species_names.index(name)] = 1
        seeds.append(seed)
        return np.array(seeds)

    def _export_results(
        self,
        best: np.ndarray,
        history: Sequence[float],
        debug: Sequence[Sequence[tuple]],
        best_details: Mapping[str, object],
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
        best_metrics = {
            case: data["metrics"] for case, data in best_details.get("cases", {}).items()
        }
        with (out_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metrics, handle, indent=2)
        summary = {
            "runtime_s": elapsed,
            "baseline_species": len(self.species_names),
            "best_species": int(best.sum()),
            "penalty": best_details.get("penalty", 0.0),
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        mechanism_name = Path(self.config.mechanism).stem
        reduced_solution = self._solution_from_genome(best)
        yaml_path = out_dir / f"{mechanism_name}_reduced.yaml"
        reduced_solution.write_yaml(str(yaml_path))
        with (out_dir / "best_details.json").open("w", encoding="utf-8") as handle:
            json.dump(_serialize_details(best_details), handle, indent=2)


def _serialize_details(details: Mapping[str, object]) -> Mapping[str, object]:
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    return convert(details)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    denom = np.linalg.norm(reference) + 1e-12
    if denom == 0.0:
        return float(np.linalg.norm(candidate))
    return float(np.linalg.norm(candidate - reference) / denom)
