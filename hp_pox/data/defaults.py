"""Default HP-POX benchmark configuration values."""

from __future__ import annotations

from typing import Dict, List, Tuple

KELVIN_OFFSET = 273.15

# Notes for Codex: residence time across the benchmark is ~15.5 s and wall
# images for Case 4 are unreliable.  These notes are mirrored from the user
# prompt so downstream scripts can expose them via CLI help if desired.

BASE_GEOMETRY = {
    "segments": [
        {"length_m": 0.305, "diameter_m": 0.50},
        {"length_m": 0.190, "diameter_m": 0.50},
        {"length_m": 2.470, "diameter_m": 0.44},
    ]
}

BASE_HEAT_LOSS = {
    "mode": "fixed_wall_temperature",
    "heat_transfer_coefficient_W_m2K": 180.0,
    "wall_temperature_profile": [
        [0.0, 133.0 + KELVIN_OFFSET],
        [0.45, 137.0 + KELVIN_OFFSET],
        [0.90, 143.0 + KELVIN_OFFSET],
        [1.35, 132.0 + KELVIN_OFFSET],
        [1.80, 141.0 + KELVIN_OFFSET],
        [2.25, 143.0 + KELVIN_OFFSET],
        [2.70, 102.0 + KELVIN_OFFSET],
    ],
}

CASE4_HEAT_LOSS = {
    "mode": "fixed_wall_temperature",
    "heat_transfer_coefficient_W_m2K": 220.0,
    "wall_temperature_profile": [
        [0.0, 140.0 + KELVIN_OFFSET],
        [0.45, 146.0 + KELVIN_OFFSET],
        [0.90, 154.0 + KELVIN_OFFSET],
        [1.35, 140.0 + KELVIN_OFFSET],
        [1.80, 149.0 + KELVIN_OFFSET],
        [2.25, 151.0 + KELVIN_OFFSET],
        [2.70, 107.0 + KELVIN_OFFSET],
    ],
}

HP_POX_DEFAULTS: Dict[str, Dict[str, object]] = {
    "Case1": {
        "pressure_bar": 50.0,
        "target_temperature_K": 1200.0 + KELVIN_OFFSET,
        "residence_time_s": 15.5,
        "friction_factor": 0.02,
        "streams": [
            {
                "name": "natural_gas",
                "mass_flow_kg_per_h": 224.07,
                "temperature_K": 359.1 + KELVIN_OFFSET,
                "composition": {
                    "CH4": 0.9597,
                    "CO2": 0.0034,
                    "C2H6": 0.0226,
                    "C3H8": 0.0046,
                    "NC4H10": 0.0005,
                    "IC4H10": 0.0006,
                    "N2": 0.0086,
                },
                "basis": "mole",
            },
            {
                "name": "oxygen",
                "mass_flow_kg_per_h": 265.37,
                "temperature_K": 240.2 + KELVIN_OFFSET,
                "composition": {"O2": 0.995, "N2": 0.005},
                "basis": "mole",
            },
            {
                "name": "steam_primary",
                "mass_flow_kg_per_h": 78.02,
                "temperature_K": 289.6 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "steam_secondary",
                "mass_flow_kg_per_h": 23.15,
                "temperature_K": 240.2 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "optical_flush",
                "mass_flow_kg_per_h": 4.72,
                "temperature_K": 25.0 + KELVIN_OFFSET,
                "composition": {"N2": 1.0},
                "basis": "mole",
            },
        ],
        "expected_syngas": {
            "H2": 0.4827,
            "CO": 0.2379,
            "CO2": 0.0419,
            "CH4": 0.0376,
            "H2O": 0.1933,
            "N2": 0.0065,
        },
        "geometry": BASE_GEOMETRY,
        "heat_loss": BASE_HEAT_LOSS,
    },
    "Case2": {
        "pressure_bar": 60.0,
        "target_temperature_K": 1200.0 + KELVIN_OFFSET,
        "residence_time_s": 15.5,
        "friction_factor": 0.02,
        "streams": [
            {
                "name": "natural_gas",
                "mass_flow_kg_per_h": 267.37,
                "temperature_K": 362.1 + KELVIN_OFFSET,
                "composition": {
                    "CH4": 0.9649,
                    "CO2": 0.0020,
                    "C2H6": 0.0191,
                    "C3H8": 0.0044,
                    "NC4H10": 0.0005,
                    "IC4H10": 0.0006,
                    "N2": 0.0085,
                },
                "basis": "mole",
            },
            {
                "name": "oxygen",
                "mass_flow_kg_per_h": 312.19,
                "temperature_K": 239.6 + KELVIN_OFFSET,
                "composition": {"O2": 0.995, "N2": 0.005},
                "basis": "mole",
            },
            {
                "name": "steam_primary",
                "mass_flow_kg_per_h": 90.01,
                "temperature_K": 299.2 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "steam_secondary",
                "mass_flow_kg_per_h": 28.63,
                "temperature_K": 239.6 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "optical_flush",
                "mass_flow_kg_per_h": 5.49,
                "temperature_K": 25.0 + KELVIN_OFFSET,
                "composition": {"N2": 1.0},
                "basis": "mole",
            },
        ],
        "expected_syngas": {
            "H2": 0.4832,
            "CO": 0.2373,
            "CO2": 0.0410,
            "CH4": 0.0417,
            "H2O": 0.1904,
            "N2": 0.0065,
        },
        "geometry": BASE_GEOMETRY,
        "heat_loss": BASE_HEAT_LOSS,
    },
    "Case3": {
        "pressure_bar": 70.0,
        "target_temperature_K": 1200.0 + KELVIN_OFFSET,
        "residence_time_s": 15.5,
        "friction_factor": 0.02,
        "streams": [
            {
                "name": "natural_gas",
                "mass_flow_kg_per_h": 314.48,
                "temperature_K": 365.3 + KELVIN_OFFSET,
                "composition": {
                    "CH4": 0.9570,
                    "CO2": 0.0042,
                    "C2H6": 0.0238,
                    "C3H8": 0.0047,
                    "NC4H10": 0.0006,
                    "IC4H10": 0.0007,
                    "N2": 0.0090,
                },
                "basis": "mole",
            },
            {
                "name": "oxygen",
                "mass_flow_kg_per_h": 355.88,
                "temperature_K": 240.6 + KELVIN_OFFSET,
                "composition": {"O2": 0.995, "N2": 0.005},
                "basis": "mole",
            },
            {
                "name": "steam_primary",
                "mass_flow_kg_per_h": 107.23,
                "temperature_K": 309.7 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "steam_secondary",
                "mass_flow_kg_per_h": 31.50,
                "temperature_K": 240.6 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "optical_flush",
                "mass_flow_kg_per_h": 5.76,
                "temperature_K": 25.0 + KELVIN_OFFSET,
                "composition": {"N2": 1.0},
                "basis": "mole",
            },
        ],
        "expected_syngas": {
            "H2": 0.4840,
            "CO": 0.2364,
            "CO2": 0.0413,
            "CH4": 0.0455,
            "H2O": 0.1864,
            "N2": 0.0063,
        },
        "geometry": BASE_GEOMETRY,
        "heat_loss": BASE_HEAT_LOSS,
    },
    "Case4": {
        "pressure_bar": 50.0,
        "target_temperature_K": 1400.0 + KELVIN_OFFSET,
        "residence_time_s": 15.5,
        "friction_factor": 0.02,
        "streams": [
            {
                "name": "natural_gas",
                "mass_flow_kg_per_h": 195.90,
                "temperature_K": 355.2 + KELVIN_OFFSET,
                "composition": {
                    "CH4": 0.9620,
                    "CO2": 0.0026,
                    "C2H6": 0.0209,
                    "C3H8": 0.0047,
                    "NC4H10": 0.0005,
                    "IC4H10": 0.0006,
                    "N2": 0.0087,
                },
                "basis": "mole",
            },
            {
                "name": "oxygen",
                "mass_flow_kg_per_h": 280.25,
                "temperature_K": 239.9 + KELVIN_OFFSET,
                "composition": {"O2": 0.995, "N2": 0.005},
                "basis": "mole",
            },
            {
                "name": "steam_primary",
                "mass_flow_kg_per_h": 64.39,
                "temperature_K": 277.5 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "steam_secondary",
                "mass_flow_kg_per_h": 22.77,
                "temperature_K": 239.9 + KELVIN_OFFSET,
                "composition": {"H2O": 1.0},
                "basis": "mole",
            },
            {
                "name": "optical_flush",
                "mass_flow_kg_per_h": 4.79,
                "temperature_K": 25.0 + KELVIN_OFFSET,
                "composition": {"N2": 1.0},
                "basis": "mole",
            },
        ],
        "expected_syngas": {
            "H2": 0.4806,
            "CO": 0.2561,
            "CO2": 0.0389,
            "CH4": 0.0006,
            "H2O": 0.2171,
            "N2": 0.0067,
        },
        "geometry": BASE_GEOMETRY,
        "heat_loss": CASE4_HEAT_LOSS,
    },
}

PLANT_GEOMETRIES: Dict[str, Dict[str, object]] = {
    "PlantA": {
        "segments": [
            {"length_m": 0.75, "diameter_m": 0.65},
            {"length_m": 2.20, "diameter_m": 0.60},
            {"length_m": 2.30, "diameter_m": 0.55},
        ],
        "volume_l": 450.0,
    },
    "PlantB": {
        "segments": [
            {"length_m": 0.50, "diameter_m": 0.50},
            {"length_m": 1.60, "diameter_m": 0.45},
            {"length_m": 1.10, "diameter_m": 0.40},
        ],
        "volume_l": 134.0,
    },
}

DEFAULT_OPERATING_ENVELOPE: Dict[str, List[Tuple[str, float, float]]] = {
    "PlantA": [
        ("pressure_bar", 40.0, 70.0),
        ("oxygen_carbon_ratio", 0.5, 0.8),
        ("steam_carbon_ratio", 0.3, 0.6),
        ("inlet_temperature_K", 650.0, 900.0),
    ],
    "PlantB": [
        ("pressure_bar", 30.0, 60.0),
        ("oxygen_carbon_ratio", 0.4, 0.75),
        ("steam_carbon_ratio", 0.2, 0.5),
        ("inlet_temperature_K", 600.0, 850.0),
    ],
}
