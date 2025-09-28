"""Command-line interface for running the GA+GNN reduction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .reduction import GAGNNReducer, ReductionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reduced mechanism using GA+GNN."
    )
    parser.add_argument(
        "--mechanism", required=True, help="Path to the full mechanism YAML file."
    )
    parser.add_argument(
        "--cases-1d",
        nargs="*",
        help="Benchmark 1-D cases to include in the fitness evaluation.",
    )
    parser.add_argument(
        "--cases-0d",
        nargs="*",
        help="Optional zero-dimensional cases for PV/IDT penalties (not yet implemented).",
    )
    parser.add_argument(
        "--gnn-seed", dest="gnn_seed", help="CSV file with GNN scores for seeding."
    )
    parser.add_argument(
        "--keep-core",
        dest="keep_core",
        action="store_true",
        help="Keep core species fixed.",
    )
    parser.add_argument(
        "--no-keep-core",
        dest="keep_core",
        action="store_false",
        help="Allow GA to drop core species.",
    )
    parser.set_defaults(keep_core=None)
    defaults = ReductionConfig.__dataclass_fields__
    parser.add_argument(
        "--pop",
        type=int,
        default=defaults["population_size"].default,
        help="Population size.",
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=defaults["generations"].default,
        help="Number of generations.",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=defaults["elitism"].default,
        help="Elitism setting for GA.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out", type=Path, default=ReductionConfig.output_dir, help="Output directory."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = ReductionConfig.__dataclass_fields__
    if args.cases_1d:
        cases_1d = tuple(case.split(":", 1)[0] for case in args.cases_1d)
    else:
        cases_1d = defaults["cases_1d"].default
    cases_0d = tuple(args.cases_0d) if args.cases_0d else defaults["cases_0d"].default
    if args.cases_0d:
        print(
            "Notice: 0-D training cases are not yet supported and will be ignored.",
        )
    keep_core = (
        args.keep_core if args.keep_core is not None else defaults["keep_core"].default
    )
    config = ReductionConfig(
        mechanism=args.mechanism,
        cases_1d=cases_1d,
        cases_0d=cases_0d,
        population_size=args.pop,
        generations=args.gens,
        elitism=args.elitism,
        seed=args.seed,
        output_dir=args.out,
        gnn_seed_path=args.gnn_seed,
        keep_core=keep_core,
    )
    reducer = GAGNNReducer(config)
    result = reducer.run()
    print(
        "Reduction complete. Best genome size: {} species".format(
            int(result.best_genome.sum())
        )
    )


if __name__ == "__main__":
    main()
