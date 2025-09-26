"""CLI driver for GA–GNN mechanism reduction tailored to HP-POX."""

from __future__ import annotations

import argparse
from pathlib import Path

from hp_pox import GAGNNReducer, ReductionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce mechanisms using GA–GNN with HP-POX KPIs.")
    parser.add_argument("--mechanism", default="data/gri30.yaml")
    parser.add_argument("--cases", nargs="*", default=["Case1", "Case4"], help="Reference cases for fidelity")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--min-species", type=int, default=25)
    parser.add_argument("--max-species", type=int, default=80)
    parser.add_argument("--gnn-seed", type=Path, help="Optional CSV with GNN species scores")
    parser.add_argument("--out", type=Path, default=Path("results/reduction"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ReductionConfig(
        mechanism=str(args.mechanism),
        cases=tuple(args.cases),
        population_size=args.population,
        generations=args.generations,
        min_species=args.min_species,
        max_species=args.max_species,
        gnn_seed_path=str(args.gnn_seed) if args.gnn_seed else None,
        output_dir=args.out,
    )
    reducer = GAGNNReducer(config)
    result = reducer.run()
    print(
        "Reduction complete. Baseline metrics stored in",
        result.output_dir / "baseline_metrics.json",
    )


if __name__ == "__main__":
    main()
