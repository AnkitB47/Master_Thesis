"""CLI utilities for exploring plasma-assisted HP-POX surrogates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from hp_pox import (
    PlugFlowOptions,
    PlugFlowSolver,
    PlasmaSurrogateConfig,
    load_case_definition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore plasma surrogate behaviour.")
    parser.add_argument("--case", default="Case1")
    parser.add_argument("--mechanism", default="data/gri30.yaml")
    parser.add_argument("--mode", choices=["thermal", "radical"], default="thermal")
    parser.add_argument("--power", type=float, default=0.0, help="Plasma power for thermal mode (W)")
    parser.add_argument("--power-range", nargs=3, type=float, metavar=("start", "stop", "num"), help="Optional sweep range for power")
    parser.add_argument("--radical-flow", type=float, default=0.0, help="Total radical molar flow (kmol/s)")
    parser.add_argument(
        "--radical-range", nargs=3, type=float, metavar=("start", "stop", "num"), help="Optional sweep for radical flow"
    )
    parser.add_argument("--radicals", default="H:0.02,O:0.01,OH:0.01", help="Radical composition string (species:frac,...)")
    parser.add_argument("--start", type=float, default=0.15, help="Injection start location (m)")
    parser.add_argument("--end", type=float, default=0.35, help="Injection end location (m)")
    parser.add_argument("--out", type=Path, default=Path("results/plasma"))
    parser.add_argument("--points", type=int, default=220)
    parser.add_argument(
        "--feed-compat",
        choices=["lump_to_propane", "lump_to_methane", "drop_and_renorm"],
        default="lump_to_propane",
        help="Policy for reconciling inlet streams with the mechanism.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = load_case_definition(args.case)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "thermal":
        sweep_values = _build_sweep(args.power, args.power_range)
        records = []
        for power in sweep_values:
            plasma = PlasmaSurrogateConfig(
                mode="thermal",
                plasma_power_W=power,
                start_position_m=args.start,
                end_position_m=args.end,
            )
            solver = PlugFlowSolver(
                args.mechanism,
                case,
                PlugFlowOptions(output_points=args.points),
                plasma=plasma,
                feed_compat_policy=args.feed_compat,
            )
            result = solver.solve()
            records.append({"plasma_power_W": power} | result.metrics)
        df = pd.DataFrame(records)
    else:
        radicals = _parse_radicals(args.radicals)
        sweep_values = _build_sweep(args.radical_flow, args.radical_range)
        records = []
        for flow in sweep_values:
            plasma = PlasmaSurrogateConfig(
                mode="radical",
                radical_injection=radicals,
                radical_molar_flow_kmol_per_s=flow,
                start_position_m=args.start,
                end_position_m=args.end,
                injection_width_m=max(args.end - args.start, 1e-3),
            )
            solver = PlugFlowSolver(
                args.mechanism,
                case,
                PlugFlowOptions(output_points=args.points),
                plasma=plasma,
                feed_compat_policy=args.feed_compat,
            )
            result = solver.solve()
            records.append({"radical_flow_kmol_s": flow} | result.metrics)
        df = pd.DataFrame(records)

    df.to_csv(out_dir / f"{args.case}_{args.mode}_sweep.csv", index=False)
    print(f"Stored sweep results in {out_dir}")


def _build_sweep(default: float, sweep: List[float] | None) -> np.ndarray:
    if sweep is None:
        return np.array([default])
    start, stop, num = sweep
    return np.linspace(start, stop, int(num))


def _parse_radicals(spec: str) -> Dict[str, float]:
    radicals: Dict[str, float] = {}
    for token in spec.split(","):
        if not token:
            continue
        name, value = token.split(":")
        radicals[name.strip()] = float(value)
    return radicals


if __name__ == "__main__":
    main()
