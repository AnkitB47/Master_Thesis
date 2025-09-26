"""CLI for running plant-scale studies using the HP-POX framework."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cantera as ct
import numpy as np
import pandas as pd

from hp_pox import (
    CaseDefinition,
    GeometryProfile,
    GeometrySegment,
    HeatLossModel,
    InletStream,
    OperatingEnvelope,
    PlugFlowOptions,
    PlugFlowSolver,
    load_case_definition,
    load_plant_definition,
)
from hp_pox.pfr import _stream_properties


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Plant A/B HP-POX applications.")
    parser.add_argument("--case", default="PlantA", choices=["PlantA", "PlantB"], help="Plant configuration")
    parser.add_argument("--baseline-case", default="Case1", help="Reference HP-POX case for stream defaults")
    parser.add_argument("--mechanism", default="data/gri30.yaml")
    parser.add_argument("--samples", type=int, default=3, help="Number of points per envelope dimension")
    parser.add_argument("--operating-envelope", type=Path, help="Optional override envelope JSON")
    parser.add_argument("--heatloss", type=Path, help="Heat-loss model override")
    parser.add_argument("--out", type=Path, default=Path("results/plant"))
    parser.add_argument("--plant-data", type=Path, help="Optional CSV with measured plant KPIs for calibration")
    parser.add_argument("--fit-U", action="store_true", help="Fit an effective U(x) profile to plant data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plant = load_plant_definition(args.case)
    base_case = load_case_definition(args.baseline_case)
    base_case = replace(base_case, geometry=plant.geometry)
    if args.heatloss:
        base_case = replace(base_case, heat_loss=_load_heat_loss(args.heatloss))
    envelope = args.operating_envelope
    if envelope:
        entries = _load_envelope(envelope)
        plant_envelope = OperatingEnvelope(entries)
    else:
        plant_envelope = plant.operating_envelope

    gas = ct.Solution(args.mechanism)
    base_ratios = _compute_base_ratios(base_case, gas)

    samples = _generate_samples(plant_envelope.entries, args.samples)
    results: List[Dict[str, float]] = []
    for sample in samples:
        case_variant = _apply_sample(base_case, sample, base_ratios)
        solver = PlugFlowSolver(args.mechanism, case_variant, PlugFlowOptions(output_points=160))
        result = solver.solve()
        record = {
            "pressure_bar": case_variant.pressure_bar,
            "oxygen_carbon_ratio": sample.get("oxygen_carbon_ratio", base_ratios["oxygen_carbon_ratio"]),
            "steam_carbon_ratio": sample.get("steam_carbon_ratio", base_ratios["steam_carbon_ratio"]),
            "inlet_temperature_K": sample.get("inlet_temperature_K", base_ratios["inlet_temperature_K"]),
        }
        record.update(result.metrics)
        results.append(record)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"{args.case}_sweep.csv", index=False)

    if args.plant_data and args.plant_data.exists():
        plant_df = pd.read_csv(args.plant_data)
        plant_df.to_csv(out_dir / f"{args.case}_plant_data.csv", index=False)

    print(f"Generated {len(results)} operating points for {args.case}.")


def _apply_sample(case: CaseDefinition, sample: Dict[str, float], base_ratios: Dict[str, float]) -> CaseDefinition:
    updated = deepcopy(case)
    if "pressure_bar" in sample:
        updated = replace(updated, pressure_bar=sample["pressure_bar"])
    o2_scale = sample.get("oxygen_carbon_ratio", base_ratios["oxygen_carbon_ratio"]) / base_ratios["oxygen_carbon_ratio"]
    steam_scale = sample.get("steam_carbon_ratio", base_ratios["steam_carbon_ratio"]) / base_ratios["steam_carbon_ratio"]
    inlet_temperature = sample.get("inlet_temperature_K", base_ratios["inlet_temperature_K"])

    new_streams: List[InletStream] = []
    for stream in updated.streams:
        mass_flow = stream.mass_flow_kg_per_h
        if "oxygen" in stream.name:
            mass_flow *= o2_scale
        if "steam" in stream.name:
            mass_flow *= steam_scale
        temperature = stream.temperature_K
        if stream.name == "natural_gas":
            temperature = inlet_temperature
        new_streams.append(
            InletStream(
                name=stream.name,
                mass_flow_kg_per_h=mass_flow,
                temperature_K=temperature,
                composition=dict(stream.composition),
                basis=stream.basis,
            )
        )
    updated = replace(updated, streams=tuple(new_streams))
    return updated


def _compute_base_ratios(case: CaseDefinition, gas: ct.Solution) -> Dict[str, float]:
    oxygen_flow = 0.0
    steam_flow = 0.0
    carbon_flow = 0.0
    for stream in case.streams:
        mass_flow, _, molar_flow = _stream_properties(stream, gas, case.pressure_Pa)
        species_names = gas.species_names
        for idx, name in enumerate(species_names):
            atoms_c = gas.n_atoms(name, "C")
            carbon_flow += molar_flow[idx] * atoms_c
            if name == "O2":
                oxygen_flow += molar_flow[idx]
            if name == "H2O":
                steam_flow += molar_flow[idx]
    if carbon_flow <= 0:
        raise ValueError("Carbon flow cannot be zero for natural-gas feed")
    return {
        "oxygen_carbon_ratio": oxygen_flow / carbon_flow,
        "steam_carbon_ratio": steam_flow / carbon_flow,
        "inlet_temperature_K": next(
            stream.temperature_K for stream in case.streams if stream.name == "natural_gas"
        ),
    }


def _generate_samples(entries: Sequence[tuple[str, float, float]], points: int) -> List[Dict[str, float]]:
    grids = []
    for key, low, high in entries:
        grids.append(np.linspace(low, high, points))
    mesh = np.meshgrid(*grids, indexing="ij")
    samples: List[Dict[str, float]] = []
    for idx in np.ndindex(*[points] * len(entries)):
        sample = {}
        for axis, (key, _, _) in enumerate(entries):
            sample[key] = float(mesh[axis][idx])
        samples.append(sample)
    return samples


def _load_heat_loss(path: Path) -> HeatLossModel:
    data = json.loads(path.read_text())
    return HeatLossModel(**data)


def _load_envelope(path: Path) -> Sequence[tuple[str, float, float]]:
    data = json.loads(path.read_text())
    return [(item["parameter"], item["min"], item["max"]) for item in data]


if __name__ == "__main__":
    main()
