"""CLI for validating the HP-POX 1-D model against benchmark cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from hp_pox import (
    CaseDefinition,
    GeometryProfile,
    GeometrySegment,
    HeatLossModel,
    PlugFlowOptions,
    PlugFlowSolver,
    PlasmaSurrogateConfig,
    load_case_definition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HP-POX reference validation simulations.")
    parser.add_argument("--case", default="Case1", help="Benchmark case identifier (Case1–Case4).")
    parser.add_argument("--mechanism", default="data/gri30.yaml", help="Path to the Cantera mechanism file.")
    parser.add_argument("--geometry", type=Path, help="Optional geometry JSON override.")
    parser.add_argument("--heatloss", type=Path, help="Optional heat-loss JSON override.")
    parser.add_argument("--operating-envelope", type=Path, help="Ignored for reference runs but parsed for completeness.")
    parser.add_argument("--plant-data", type=Path, help="Optional CSV with experimental plant KPIs for comparison.")
    parser.add_argument("--friction-factor", type=float, help="Override Darcy friction factor.")
    parser.add_argument("--plasma", choices=["none", "surrogate_thermal", "surrogate_radical"], default="none")
    parser.add_argument("--plasma-power", type=float, default=0.0, help="Plasma power for thermal surrogate (W).")
    parser.add_argument("--plasma-start", type=float, default=0.2, help="Start position of the plasma zone (m).")
    parser.add_argument("--plasma-end", type=float, default=0.4, help="End position of the plasma zone (m).")
    parser.add_argument(
        "--plasma-radicals",
        default="H:0.02,O:0.01,OH:0.01",
        help="Comma-separated radical mole fractions for radical surrogate.",
    )
    parser.add_argument(
        "--plasma-radical-flow",
        type=float,
        default=0.0,
        help="Total radical molar flow rate for surrogate injections (kmol/s).",
    )
    parser.add_argument("--out", type=Path, default=Path("results/reference"), help="Output directory.")
    parser.add_argument("--points", type=int, default=240, help="Number of axial output points.")
    parser.add_argument("--ignition-method", choices=["dTdx", "temperature"], default="dTdx")
    parser.add_argument("--ignition-threshold", type=float, default=150.0, help="Threshold for ignition metric.")
    parser.add_argument("--ignition-temperature", type=float, default=1300.0, help="Ignition threshold temperature (K).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = load_case_definition(args.case)
    if args.geometry:
        case.geometry = _load_geometry(args.geometry)
    if args.heatloss:
        case.heat_loss = _load_heat_loss(args.heatloss)
    if args.friction_factor is not None:
        case = replace(case, friction_factor=args.friction_factor)
    plasma = None
    if args.plasma != "none":
        plasma = _build_plasma_config(args)
    options = PlugFlowOptions(
        output_points=args.points,
        ignition_method=args.ignition_method,
        ignition_threshold=args.ignition_threshold,
        ignition_temperature_K=args.ignition_temperature,
    )
    solver = PlugFlowSolver(args.mechanism, case, options=options, plasma=plasma)
    result = solver.solve()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    df = result.to_dataframe()
    df.to_csv(out_dir / f"{case.name}_profiles.csv", index=False)
    with (out_dir / f"{case.name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2)

    _plot_profiles(result, out_dir / f"{case.name}_profiles.png")

    if args.plant_data and args.plant_data.exists():
        plant_df = pd.read_csv(args.plant_data)
        plant_df.to_csv(out_dir / "plant_reference.csv", index=False)

    print(f"Simulation for {case.name} complete. Metrics: {result.metrics}")


def _load_geometry(path: Path) -> GeometryProfile:
    data = json.loads(path.read_text())
    segments = [GeometrySegment(**segment) for segment in data["segments"]]
    return GeometryProfile(segments)


def _load_heat_loss(path: Path) -> HeatLossModel:
    data = json.loads(path.read_text())
    return HeatLossModel(**data)


def _build_plasma_config(args: argparse.Namespace) -> PlasmaSurrogateConfig:
    if args.plasma == "surrogate_thermal":
        return PlasmaSurrogateConfig(
            mode="thermal",
            start_position_m=args.plasma_start,
            end_position_m=args.plasma_end,
            plasma_power_W=args.plasma_power,
        )
    radicals = _parse_radicals(args.plasma_radicals)
    return PlasmaSurrogateConfig(
        mode="radical",
        start_position_m=args.plasma_start,
        end_position_m=args.plasma_end,
        radical_injection=radicals,
        radical_molar_flow_kmol_per_s=args.plasma_radical_flow,
        injection_width_m=max(args.plasma_end - args.plasma_start, 1e-3),
    )


def _parse_radicals(spec: str) -> Dict[str, float]:
    radicals: Dict[str, float] = {}
    for token in spec.split(","):
        if not token:
            continue
        name, value = token.split(":")
        radicals[name.strip()] = float(value)
    return radicals


def _plot_profiles(result, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax[0].plot(result.positions_m, result.temperature_K)
    ax[0].set_xlabel("Axial position (m)")
    ax[0].set_ylabel("Temperature (K)")
    ax[0].set_title("Axial temperature")

    key_species = [sp for sp in ("CH4", "CO", "H2", "CO2", "H2O") if sp in result.mole_fractions]
    for name in key_species:
        ax[1].plot(result.positions_m, result.mole_fractions[name], label=name)
    ax[1].set_xlabel("Axial position (m)")
    ax[1].set_ylabel("Mole fraction")
    ax[1].legend()
    ax[1].set_title("Species profiles")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
