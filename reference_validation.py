"""CLI for validating the HP-POX 1-D model against benchmark cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import numpy as np
import cantera as ct

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
from hp_pox.plots import (
    plot_progress_variable_overlay,
    plot_ignition_positions,
    plot_kpi_summary,
    plot_placeholder,
    plot_profile_overlay,
    plot_profile_residuals,
    plot_timescales_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HP-POX reference validation simulations."
    )
    parser.add_argument(
        "--case", default="Case1", help="Benchmark case identifier (Case1–Case4)."
    )
    parser.add_argument(
        "--mechanism",
        default="data/gri30.yaml",
        help="Path to the Cantera mechanism file.",
    )
    parser.add_argument(
        "--geometry", type=Path, help="Optional geometry JSON override."
    )
    parser.add_argument(
        "--heatloss", type=Path, help="Optional heat-loss JSON override."
    )
    parser.add_argument(
        "--operating-envelope",
        type=Path,
        help="Ignored for reference runs but parsed for completeness.",
    )
    parser.add_argument(
        "--plant-data",
        type=Path,
        help="Optional CSV with experimental plant KPIs for comparison.",
    )
    parser.add_argument(
        "--friction-factor", type=float, help="Override Darcy friction factor."
    )
    parser.add_argument(
        "--plasma",
        choices=["none", "surrogate_thermal", "surrogate_radical"],
        default="none",
    )
    parser.add_argument(
        "--plasma-power",
        type=float,
        default=0.0,
        help="Plasma power for thermal surrogate (W).",
    )
    parser.add_argument(
        "--plasma-start",
        type=float,
        default=0.2,
        help="Start position of the plasma zone (m).",
    )
    parser.add_argument(
        "--plasma-end",
        type=float,
        default=0.4,
        help="End position of the plasma zone (m).",
    )
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
    parser.add_argument(
        "--out", type=Path, default=Path("results/reference"), help="Output directory."
    )
    parser.add_argument(
        "--points", type=int, default=240, help="Number of axial output points."
    )
    parser.add_argument(
        "--ignition-method", choices=["dTdx", "temperature"], default="dTdx"
    )
    parser.add_argument(
        "--ignition-threshold",
        type=float,
        default=150.0,
        help="Threshold for ignition metric.",
    )
    parser.add_argument(
        "--ignition-temperature",
        type=float,
        default=1300.0,
        help="Ignition threshold temperature (K).",
    )
    parser.add_argument(
        "--include-wall-heat-loss",
        dest="include_wall_heat_loss",
        action="store_true",
        help="Enable wall heat losses in the plug-flow solver.",
    )
    parser.add_argument(
        "--no-wall-heat-loss",
        dest="include_wall_heat_loss",
        action="store_false",
        help="Disable wall heat losses in the plug-flow solver.",
    )
    parser.set_defaults(include_wall_heat_loss=None)
    parser.add_argument(
        "--feed-compat",
        choices=["lump_to_propane", "lump_to_methane", "drop_and_renorm"],
        default="lump_to_propane",
        help="Policy for reconciling feed compositions with the reaction mechanism.",
    )
    parser.add_argument(
        "--compare-mechanisms",
        nargs=2,
        metavar=("FULL", "REDUCED"),
        help="Compare two mechanisms and export overlays.",
    )
    parser.add_argument(
        "--kpi-sweep",
        action="store_true",
        help="Run KPI sweeps versus pressure and outlet temperature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    base_case = load_case_definition(args.case)
    if args.geometry:
        base_case.geometry = _load_geometry(args.geometry)
    if args.heatloss:
        base_case.heat_loss = _load_heat_loss(args.heatloss)
    if args.friction_factor is not None:
        base_case = replace(base_case, friction_factor=args.friction_factor)
    plasma = _build_plasma_config(args) if args.plasma != "none" else None

    if args.compare_mechanisms:
        full_mech, reduced_mech = (
            _validate_mechanism_path(args.compare_mechanisms[0]),
            _validate_mechanism_path(args.compare_mechanisms[1]),
        )
        full_result = _run_simulation(
            full_mech,
            base_case,
            args,
            plasma=plasma,
        )
        reduced_result = _run_simulation(
            reduced_mech,
            base_case,
            args,
            plasma=plasma,
        )
        _export_comparison(
            full_mech,
            reduced_mech,
            full_result,
            reduced_result,
            out_dir,
        )
    else:
        result = _run_simulation(
            args.mechanism,
            base_case,
            args,
            plasma=plasma,
        )
        _export_single_result(args.mechanism, result, out_dir)

    if args.kpi_sweep:
        mech = args.compare_mechanisms[1] if args.compare_mechanisms else args.mechanism
        _run_kpi_sweep(mech, base_case, args, plasma, out_dir)

    if args.plant_data and args.plant_data.exists():
        plant_df = pd.read_csv(args.plant_data)
        plant_df.to_csv(out_dir / "plant_reference.csv", index=False)


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


def _solver_options(args: argparse.Namespace) -> PlugFlowOptions:
    include_wall_heat = PlugFlowOptions().include_wall_heat_loss
    if args.include_wall_heat_loss is not None:
        include_wall_heat = args.include_wall_heat_loss
    return PlugFlowOptions(
        output_points=args.points,
        ignition_method=args.ignition_method,
        ignition_threshold=args.ignition_threshold,
        ignition_temperature_K=args.ignition_temperature,
        include_wall_heat_loss=include_wall_heat,
    )


def _run_simulation(
    mechanism: str,
    case: CaseDefinition,
    args: argparse.Namespace,
    plasma: PlasmaSurrogateConfig | None,
):
    options = _solver_options(args)
    solver = PlugFlowSolver(
        mechanism,
        case,
        options=options,
        plasma=plasma,
        feed_compat_policy=args.feed_compat,
    )
    return solver.solve()


def _export_single_result(mechanism: str, result, out_dir: Path) -> None:
    df = result.to_dataframe()
    df.to_csv(out_dir / "profiles.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2)
    key_species = [
        sp for sp in ("CH4", "CO", "H2", "CO2", "H2O") if sp in result.mole_fractions
    ]
    plot_profile_overlay(
        result.positions_m,
        result.temperature_K,
        result.mole_fractions,
        temperature_reduced=None,
        species_reduced=None,
        species=key_species,
        out_path=out_dir / "profiles.png",
        ignition_full=result.metrics.get("ignition_position_m"),
    )
    plot_placeholder(
        "Residuals require comparison run", out_dir / "profiles_residual.png"
    )
    plot_ignition_positions(
        result.metrics.get("ignition_position_m"),
        None,
        out_dir / "ignition_vs_length.png",
    )
    plot_placeholder(
        "Ignition delay unavailable", out_dir / "ignition_delay_vs_length.png"
    )
    plot_placeholder("Run --kpi-sweep to populate", out_dir / "kpis.png")


def _export_comparison(
    full_mech: str,
    reduced_mech: str,
    full_result,
    reduced_result,
    out_dir: Path,
) -> None:
    species = [
        sp
        for sp in ("CH4", "CO", "H2", "CO2", "H2O")
        if sp in full_result.mole_fractions or sp in reduced_result.mole_fractions
    ]
    df = pd.DataFrame(
        {"x_m": full_result.positions_m, "T_full": full_result.temperature_K}
    )
    df["T_reduced"] = reduced_result.temperature_K
    for specie in species:
        df[f"X_{specie}_full"] = full_result.mole_fractions.get(specie, 0.0)
        df[f"X_{specie}_reduced"] = reduced_result.mole_fractions.get(specie, 0.0)
    df.to_csv(out_dir / "profiles.csv", index=False)
    metrics_payload = {
        "full": full_result.metrics,
        "reduced": reduced_result.metrics,
        "mechanisms": {"full": full_mech, "reduced": reduced_mech},
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    plot_profile_overlay(
        full_result.positions_m,
        full_result.temperature_K,
        full_result.mole_fractions,
        reduced_result.temperature_K,
        reduced_result.mole_fractions,
        species,
        out_dir / "overlay_profiles.png",
        ignition_full=full_result.metrics.get("ignition_position_m"),
        ignition_reduced=reduced_result.metrics.get("ignition_position_m"),
    )
    residuals = {}
    for specie in species:
        if (
            specie in full_result.mole_fractions
            and specie in reduced_result.mole_fractions
        ):
            residuals[specie] = np.asarray(
                reduced_result.mole_fractions[specie]
            ) - np.asarray(full_result.mole_fractions[specie])
    if residuals:
        plot_profile_residuals(
            full_result.positions_m, residuals, out_dir / "profiles_residual.png"
        )
    else:
        plot_placeholder(
            "No overlapping species for residuals", out_dir / "profiles_residual.png"
        )
    plot_ignition_positions(
        full_result.metrics.get("ignition_position_m"),
        reduced_result.metrics.get("ignition_position_m"),
        out_dir / "ignition_vs_length.png",
        labels=(Path(full_mech).name, Path(reduced_mech).name),
    )
    plot_placeholder(
        "Ignition delay requires transient data",
        out_dir / "ignition_delay_vs_length.png",
    )
    kpi_rows = []
    for label, mech, result in (
        ("full", full_mech, full_result),
        ("reduced", reduced_mech, reduced_result),
    ):
        row = {"label": label, "mechanism": mech}
        row.update(result.metrics)
        row.update(_compute_kpis(result))
        kpi_rows.append(row)
    pd.DataFrame(kpi_rows).to_csv(out_dir / "kpis.csv", index=False)
    plot_placeholder("See kpis.csv for KPI comparison", out_dir / "kpis.png")
    pv_full = _extract_progress_variable(full_result)
    pv_reduced = _extract_progress_variable(reduced_result)
    if pv_full and pv_reduced:
        plot_progress_variable_overlay(
            pv_full[0],
            pv_full[1],
            pv_reduced[0],
            pv_reduced[1],
            out_dir / "pv_overlay.png",
        )
    else:
        plot_placeholder(
            "Progress variable diagnostics unavailable", out_dir / "pv_overlay.png"
        )
    times_full = _extract_timescales(full_result)
    times_reduced = _extract_timescales(reduced_result)
    if times_full and times_reduced:
        plot_timescales_overlay(
            times_full[0],
            times_full[1],
            times_reduced[0],
            times_reduced[1],
            out_dir / "timescales.png",
        )
    else:
        plot_placeholder(
            "Timescale diagnostics unavailable", out_dir / "timescales.png"
        )


def _run_kpi_sweep(
    mechanism: str,
    case: CaseDefinition,
    args: argparse.Namespace,
    plasma: PlasmaSurrogateConfig | None,
    out_dir: Path,
) -> None:
    pressures_bar = [50.0, 60.0, 70.0]
    temperatures_K = [1473.15, 1673.15]  # 1200 and 1400 C
    results = []
    for pressure in pressures_bar:
        for target_T in temperatures_K:
            modified_case = replace(
                case, pressure_bar=pressure, target_temperature_K=target_T
            )
            sim_result = _run_simulation(mechanism, modified_case, args, plasma)
            metrics = _compute_kpis(sim_result)
            metrics.update({"pressure_bar": pressure, "target_T_K": target_T})
            results.append(metrics)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "kpis.csv", index=False)
    if not df.empty:
        summary_values = {
            "HC conversion": list(df["hc_conversion"]),
            "Cold gas efficiency": list(df["cold_gas_efficiency"]),
            "(CO+H2)/NG LHV": list(df["lhv_ratio"]),
        }
        plot_kpi_summary(
            range(len(df)), summary_values, out_dir / "kpis.png", "Sweep index"
        )
        for sweep, label in [
            ("pressure_bar", "Pressure (bar)"),
            ("target_T_K", "Outlet temperature (K)"),
        ]:
            grouped = df.groupby(sweep)
            positions = list(grouped.groups.keys())
            values = {
                "HC conversion": [
                    group["hc_conversion"].mean() for _, group in grouped
                ],
                "Cold gas efficiency": [
                    group["cold_gas_efficiency"].mean() for _, group in grouped
                ],
                "(CO+H2)/NG LHV": [group["lhv_ratio"].mean() for _, group in grouped],
            }
            plot_kpi_summary(positions, values, out_dir / f"kpis_{sweep}.png", label)
    else:
        plot_placeholder("KPI sweep produced no data", out_dir / "kpis.png")


def _compute_kpis(result) -> Dict[str, float]:
    species = result.molar_flows
    outlet = {name: values[-1] for name, values in species.items()}
    inlet = {name: values[0] for name, values in species.items()}
    hc_in = inlet.get("CH4", 0.0)
    hc_out = outlet.get("CH4", 0.0)
    conversion = 0.0
    if hc_in > 0:
        conversion = 1.0 - hc_out / hc_in
    lhv = {"CH4": 802.3e3, "H2": 241.8e3, "CO": 283.0e3}
    fuel_in = sum(inlet.get(sp, 0.0) * lhv.get(sp, 0.0) for sp in lhv)
    syngas_out = sum(outlet.get(sp, 0.0) * lhv.get(sp, 0.0) for sp in ("CO", "H2"))
    cold_gas_eff = syngas_out / fuel_in if fuel_in > 0 else 0.0
    lhv_ratio = syngas_out / fuel_in if fuel_in > 0 else 0.0
    return {
        "hc_conversion": conversion,
        "cold_gas_efficiency": cold_gas_eff,
        "lhv_ratio": lhv_ratio,
    }


def _validate_mechanism_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.exists():
        raise SystemExit(f"Mechanism file not found: {path}")
    try:
        ct.Solution(str(candidate))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to load mechanism '{path}': {exc}") from exc
    return str(candidate)


def _extract_progress_variable(result) -> tuple[np.ndarray, np.ndarray] | None:
    meta = (
        result.metadata.get("progress_variable")
        if hasattr(result, "metadata")
        else None
    )
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        values = meta.get("values")
        positions = meta.get("positions") or meta.get("time")
    elif isinstance(meta, (tuple, list)) and len(meta) == 2:
        positions, values = meta
    else:
        return None
    if values is None:
        return None
    values_arr = np.asarray(values)
    if positions is None:
        positions_arr = np.linspace(0.0, 1.0, values_arr.size)
    else:
        positions_arr = np.asarray(positions)
    if positions_arr.size != values_arr.size:
        return None
    return positions_arr, values_arr


def _extract_timescales(result) -> tuple[np.ndarray, Dict[str, np.ndarray]] | None:
    meta = result.metadata.get("timescales") if hasattr(result, "metadata") else None
    if meta is None or not isinstance(meta, Mapping):
        return None
    time = meta.get("time") or meta.get("positions")
    if time is None:
        return None
    time_arr = np.asarray(time)
    series: Dict[str, np.ndarray] = {}
    for key, value in meta.items():
        if key in {"time", "positions"}:
            continue
        arr = np.asarray(value)
        if arr.shape == time_arr.shape:
            series[key] = arr
    if not series:
        return None
    return time_arr, series


if __name__ == "__main__":
    main()
