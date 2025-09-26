"""Dataclasses and loaders for HP-POX configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import json
import math

from .data.defaults import (
    DEFAULT_OPERATING_ENVELOPE,
    HP_POX_DEFAULTS,
    PLANT_GEOMETRIES,
)


@dataclass
class InletStream:
    """Definition of a single inlet stream."""

    name: str
    mass_flow_kg_per_h: float
    temperature_K: float
    composition: Mapping[str, float]
    basis: str = "mole"

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "mass_flow_kg_per_h": self.mass_flow_kg_per_h,
            "temperature_K": self.temperature_K,
            "composition": dict(self.composition),
            "basis": self.basis,
        }


@dataclass
class GeometrySegment:
    length_m: float
    diameter_m: float


@dataclass
class GeometryProfile:
    segments: Sequence[GeometrySegment]

    @property
    def total_length(self) -> float:
        return float(sum(seg.length_m for seg in self.segments))

    def area_at(self, position: float) -> float:
        seg = self._segment_at(position)
        radius = seg.diameter_m * 0.5
        return math.pi * radius * radius

    def perimeter_at(self, position: float) -> float:
        seg = self._segment_at(position)
        return math.pi * seg.diameter_m

    def hydraulic_diameter_at(self, position: float) -> float:
        seg = self._segment_at(position)
        return seg.diameter_m

    def _segment_at(self, position: float) -> GeometrySegment:
        if position <= 0:
            return self.segments[0]
        x = 0.0
        for seg in self.segments:
            x += seg.length_m
            if position <= x + 1e-12:
                return seg
        return self.segments[-1]


@dataclass
class HeatLossModel:
    mode: str
    heat_transfer_coefficient_W_m2K: float | None = None
    wall_temperature_profile: Sequence[Sequence[float]] = field(default_factory=list)
    u_profile: Sequence[Sequence[float]] = field(default_factory=list)

    def wall_temperature(self, position: float) -> float | None:
        if not self.wall_temperature_profile:
            return None
        xs, values = zip(*self.wall_temperature_profile)
        return _interp_clamped(xs, values, position)

    def u_value(self, position: float) -> float | None:
        if self.mode == "u_profile" and self.u_profile:
            xs, values = zip(*self.u_profile)
            return _interp_clamped(xs, values, position)
        return self.heat_transfer_coefficient_W_m2K


@dataclass
class OperatingEnvelope:
    entries: Sequence[tuple[str, float, float]]


@dataclass
class CaseDefinition:
    name: str
    pressure_bar: float
    target_temperature_K: float
    residence_time_s: float
    friction_factor: float
    streams: Sequence[InletStream]
    geometry: GeometryProfile
    heat_loss: HeatLossModel
    expected_syngas: Mapping[str, float] | None = None

    @property
    def pressure_Pa(self) -> float:
        return self.pressure_bar * 1e5


@dataclass
class PlantDefinition:
    name: str
    geometry: GeometryProfile
    operating_envelope: OperatingEnvelope


def load_case_definition(case: str | Path | Mapping[str, object]) -> CaseDefinition:
    """Load a case definition by name, path, or mapping."""

    if isinstance(case, Mapping):
        data = dict(case)
    else:
        data = _load_case_mapping(case)
    name = data.get("name", case if isinstance(case, str) else "custom")
    streams = [InletStream(**stream) for stream in data["streams"]]
    geometry = GeometryProfile(
        [GeometrySegment(**seg) for seg in data["geometry"]["segments"]]
    )
    heat_loss = HeatLossModel(**data.get("heat_loss", {"mode": "adiabatic"}))
    return CaseDefinition(
        name=str(name),
        pressure_bar=float(data["pressure_bar"]),
        target_temperature_K=float(data["target_temperature_K"]),
        residence_time_s=float(data["residence_time_s"]),
        friction_factor=float(data.get("friction_factor", 0.0)),
        streams=streams,
        geometry=geometry,
        heat_loss=heat_loss,
        expected_syngas=data.get("expected_syngas"),
    )


def load_plant_definition(name: str) -> PlantDefinition:
    geometry = GeometryProfile(
        [GeometrySegment(**seg) for seg in PLANT_GEOMETRIES[name]["segments"]]
    )
    envelope = OperatingEnvelope(DEFAULT_OPERATING_ENVELOPE[name])
    return PlantDefinition(name=name, geometry=geometry, operating_envelope=envelope)


def _load_case_mapping(case: str | Path) -> MutableMapping[str, object]:
    if isinstance(case, str) and case in HP_POX_DEFAULTS:
        from copy import deepcopy

        return deepcopy(HP_POX_DEFAULTS[case]) | {"name": case}
    path = Path(case)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "name" not in data:
        data["name"] = path.stem
    return data


def _interp_clamped(xs: Iterable[float], values: Iterable[float], position: float) -> float:
    xs_list = list(xs)
    vals_list = list(values)
    if position <= xs_list[0]:
        return vals_list[0]
    if position >= xs_list[-1]:
        return vals_list[-1]
    for i in range(1, len(xs_list)):
        if position < xs_list[i]:
            x0, x1 = xs_list[i - 1], xs_list[i]
            v0, v1 = vals_list[i - 1], vals_list[i]
            weight = (position - x0) / (x1 - x0)
            return v0 + weight * (v1 - v0)
    return vals_list[-1]
