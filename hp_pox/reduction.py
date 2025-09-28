"""GA-GNN reduction pipeline targeting HP-POX KPIs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import json
import logging
import time

import numpy as np
import pandas as pd

import cantera as ct

from metaheuristics.ga import GAOptions, run_ga

from .configuration import load_case_definition
from .plasma import PlasmaSurrogateConfig
from .pfr import PlugFlowOptions, PlugFlowResult, PlugFlowSolver


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
            case: data["metrics"]
            for case, data in best_details.get("cases", {}).items()
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
    def _evaluate_individual(
        self, genome: np.ndarray
    ) -> tuple[float, Dict[str, object]]:
        try:
            details = self._solve_for_genome(genome)
        except Exception as exc:  # noqa: BLE001
            return -1e6, {"error": str(exc)}
        score = -details.get("total_error", 1e6)
        return score, details

    def _solve_for_genome(self, genome: np.ndarray) -> Dict[str, object]:
        active_species = [
            name for i, name in enumerate(self.species_names) if genome[i] > 0.5
        ]
        missing = self.required_species.difference(active_species)
        if missing:
            raise ValueError(f"Genome removed required species: {sorted(missing)}")
        reduced_gas = self._solution_from_genome(genome)
        reactions = list(reduced_gas.reactions())
        if not reactions:
            raise ValueError("Genome eliminated all reactions")
        penalty = self._reaction_penalty(reactions)
        case_results, case_error = self._metrics_for_genome(reduced_gas, active_species)
        total_error = penalty + case_error
        return {
            "cases": case_results,
            "penalty": penalty,
            "total_error": total_error,
            "active_species": active_species,
            "reactions": len(reactions),
        }

    def _compare_profiles(
        self, case_name: str, candidate: PlugFlowResult
    ) -> Dict[str, object]:
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
                ignition_penalty = abs(candidate_ign - baseline_ign) / max(
                    baseline_ign, 1e-6
                )
        total += ignition_penalty
        return {
            "temperature": temp_errors,
            "species": species_errors,
            "ignition_penalty": ignition_penalty,
            "total": total,
        }

    def _region_masks(
        self, xs: np.ndarray, ignition: float | None
    ) -> Dict[str, np.ndarray]:
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

    def _reaction_penalty(self, reactions: Sequence[ct.Reaction]) -> float:
        equations = {getattr(reaction, "equation", "") for reaction in reactions}
        missing = [
            pattern
            for pattern, matches in self.required_reaction_patterns.items()
            if matches and not (equations & matches)
        ]
        return float(len(missing)) * 10.0

    def _compute_reference_profiles(self) -> Dict[str, PlugFlowResult]:
        profiles: Dict[str, PlugFlowResult] = {}
        for case in self.reference_cases:
            solver = self._reactive_solver(self.mechanism, case)
            profiles[case.name] = solver.solve()
        return profiles

    def _collect_baseline_metrics(self) -> Dict[str, Mapping[str, float]]:
        return {
            name: result.metrics for name, result in self.reference_profiles.items()
        }

    def _determine_fixed_species(self) -> Sequence[str]:
        essential = set(self.required_species)
        if self.config.keep_core:
            essential.update({"O2", "N2"})
        for case in self.reference_cases:
            for stream in case.streams:
                essential.update(stream.composition.keys())
        return [name for name in essential if name in self.species_names]

    def _metrics_for_genome(
        self, gas: ct.Solution, active_species: Sequence[str]
    ) -> tuple[Dict[str, Dict[str, object]], float]:
        case_results: Dict[str, Dict[str, object]] = {}
        total_error = 0.0
        for case in self.reference_cases:
            solver = self._reactive_solver(gas, case)
            result = solver.solve()
            errors = self._compare_profiles(case.name, result)
            total_error += errors["total"]
            case_results[case.name] = {
                "metrics": result.metrics,
                "errors": errors,
                "species": list(active_species),
            }
        return case_results, total_error

    def _reactive_solver(
        self, mechanism: ct.Solution | str | Path, case
    ) -> PlugFlowSolver:
        mech_obj = mechanism
        if isinstance(mechanism, ct.Solution):
            mech_obj = deepcopy(mechanism)
        options = PlugFlowOptions(
            output_points=self.config.sample_points,
            ignition_method="temperature",
            ignition_temperature_K=1300.0,
        )
        plasma = PlasmaSurrogateConfig(
            mode="thermal",
            start_position_m=0.05,
            end_position_m=0.25,
            plasma_power_W=2.5e5,
        )
        return PlugFlowSolver(
            mech_obj,
            case,
            options=options,
            plasma=plasma,
        )

    def _solution_from_genome(self, genome: np.ndarray) -> ct.Solution:
        allowed_names = [
            name for i, name in enumerate(self.species_names) if genome[i] > 0.5
        ]
        if not allowed_names:
            raise ValueError("No species selected")

        species_objects = [
            deepcopy(sp) for sp in self.mechanism.species() if sp.name in allowed_names
        ]
        allowed = {sp.name for sp in species_objects}

        reactions: List[ct.Reaction] = []
        for rxn in self.mechanism.reactions():
            reactant_set = set(rxn.reactants)
            product_set = set(rxn.products)
            if not (reactant_set.issubset(allowed) and product_set.issubset(allowed)):
                continue
            r2 = deepcopy(rxn)
            if hasattr(r2, "third_body") and r2.third_body is not None:
                efficiencies = r2.third_body.efficiencies or {}
                r2.third_body.efficiencies = {
                    sp: eff for sp, eff in efficiencies.items() if sp in allowed
                }
            reactions.append(r2)

        gas_red = ct.Solution(
            thermo=self.mechanism.thermo.model,
            kinetics=self.mechanism.kinetics.model,
            species=species_objects,
            reactions=reactions,
        )
        return gas_red

    def _load_seed_population(self) -> np.ndarray:
        if self.config.gnn_seed_path is None:
            return None
        path = Path(self.config.gnn_seed_path)
        if not path.exists():
            return None
        scores = pd.read_csv(path)
        species_to_idx = {name: i for i, name in enumerate(self.species_names)}
        ranked = sorted(
            (
                (species_to_idx[row["species"]], row["score"])
                for _, row in scores.iterrows()
                if row["species"] in species_to_idx
            ),
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
            case: data["metrics"]
            for case, data in best_details.get("cases", {}).items()
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
        try:
            reduced_solution.write_yaml(str(yaml_path), header="# Reduced by GA+GNN")
        except Exception as exc:  # noqa: BLE001
            import yaml

            doc = {
                "phases": [
                    {
                        "name": "gas",
                        "thermo": reduced_solution.thermo.model,
                        "kinetics": reduced_solution.kinetics.model,
                        "elements": [e.symbol for e in self.mechanism.elements()],
                        "species": [sp.name for sp in reduced_solution.species()],
                        "state": {"T": 300.0, "P": "1 atm"},
                    }
                ],
                "species": [
                    yaml.safe_load(sp.to_yaml()) for sp in reduced_solution.species()
                ],
                "reactions": [
                    yaml.safe_load(r.to_yaml()) for r in reduced_solution.reactions()
                ],
            }
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(doc, handle, sort_keys=False)
            self.logger.warning("write_yaml failed (%s); used manual YAML export", exc)
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
