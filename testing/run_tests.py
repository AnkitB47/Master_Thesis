"""CLI entry point for running the reduction testing pipeline."""
from __future__ import annotations

import argparse
import os

from testing.pipeline import full_pipeline


def main() -> None:
    """Parse command line arguments and run the pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism",
        default=os.path.join("data", "gri30.yaml"),
        help="Path to mechanism YAML file",
    )
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of time steps in the reactor simulation",
    )
    parser.add_argument(
        "--tf",
        type=float,
        default=1.0,
        help="Final time of the reactor simulation",
    )
    args = parser.parse_args()
    
    full_pipeline(args.mechanism, args.out, steps=args.steps, tf=args.tf)
    print(f"Pipeline complete. Results written to '{args.out}'.")


if __name__ == "__main__":
    main()
