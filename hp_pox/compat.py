"""Utilities for reconciling HP-POX feed compositions with mechanisms."""

from __future__ import annotations

import re
from typing import Dict, Mapping

import cantera as ct

from .configuration import load_case_definition

__all__ = [
    "reconcile_feed_with_mechanism",
    "DEFAULT_FEED_SPECIES",
    "DEFAULT_MISSING_WITH_GRI30",
]

_POLICIES = {"lump_to_propane", "lump_to_methane", "drop_and_renorm"}
_HEAVY_C_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Default-case bookkeeping
# ---------------------------------------------------------------------------

def _gather_default_feed_species() -> set[str]:
    species: set[str] = set()
    for case_name in ("Case1", "Case2", "Case3", "Case4"):
        try:
            case = load_case_definition(case_name)
        except Exception:
            continue
        for stream in case.streams:
            species.update(stream.composition.keys())
    return species


DEFAULT_FEED_SPECIES = _gather_default_feed_species()


def _scan_defaults_against_gri30() -> list[str]:
    try:
        gas = ct.Solution("data/gri30.yaml")
    except Exception:
        return []
    mechanism_species = set(gas.species_names)
    return sorted(sp for sp in DEFAULT_FEED_SPECIES if sp not in mechanism_species)


DEFAULT_MISSING_WITH_GRI30 = _scan_defaults_against_gri30()


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def reconcile_feed_with_mechanism(
    gas: ct.Solution,
    composition: Mapping[str, float],
    policy: str = "lump_to_propane",
    *,
    return_mapping: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], Dict[str, str]]:
    """Return a mechanism-compatible composition mapping.

    Parameters
    ----------
    gas:
        Cantera ``Solution`` object representing the mechanism.
    composition:
        Original feed composition mapping.
    policy:
        Strategy for reconciling species that are missing from the mechanism.
    return_mapping:
        When ``True`` the function also returns a mapping describing how
        missing species were lumped into available surrogate species.  The
        mapping keys are the missing species and the values are the surrogate
        species selected for them.

    Returns
    -------
    Dict[str, float] or tuple
        A mechanism-compatible composition.  When ``return_mapping`` is set,
        a tuple of ``(composition, mapping)`` is returned.
    """

    if policy not in _POLICIES:
        raise ValueError(f"Unknown feed compatibility policy: {policy}")

    mechanism_species = set(gas.species_names)
    cleaned: Dict[str, float] = {}
    missing_heavy = 0.0
    missing_other = 0.0
    reconciliation: Dict[str, str] = {}

    for name, value in composition.items():
        frac = float(value)
        if frac <= 0.0:
            continue
        if name in mechanism_species:
            cleaned[name] = cleaned.get(name, 0.0) + frac
        else:
            carbon = _estimate_carbon_count(name)
            if carbon is not None and carbon >= _HEAVY_C_THRESHOLD:
                missing_heavy += frac
            else:
                missing_other += frac

    if policy == "lump_to_propane":
        if missing_heavy > 0.0:
            sink = _select_propane_sink(mechanism_species)
            cleaned[sink] = cleaned.get(sink, 0.0) + missing_heavy
            for name, value in composition.items():
                if value > 0 and name not in mechanism_species:
                    carbon = _estimate_carbon_count(name)
                    if carbon is not None and carbon >= _HEAVY_C_THRESHOLD:
                        reconciliation[name] = sink
        if missing_other > 0.0:
            sink = _select_methane_sink(mechanism_species)
            cleaned[sink] = cleaned.get(sink, 0.0) + missing_other
            for name, value in composition.items():
                if value > 0 and name not in mechanism_species:
                    carbon = _estimate_carbon_count(name)
                    if carbon is None or carbon < _HEAVY_C_THRESHOLD:
                        reconciliation[name] = sink
    elif policy == "lump_to_methane":
        total_missing = missing_heavy + missing_other
        if total_missing > 0.0:
            sink = _select_methane_sink(mechanism_species)
            cleaned[sink] = cleaned.get(sink, 0.0) + total_missing
            for name, value in composition.items():
                if value > 0 and name not in mechanism_species:
                    reconciliation[name] = sink
    elif policy == "drop_and_renorm":
        pass

    total = sum(cleaned.values())
    if total <= 0.0:
        missing_total = missing_heavy + missing_other
        raise ValueError(
            "Resulting composition is empty after applying compatibility policy; "
            f"dropped fraction={missing_total:.6f}."
        )

    reconciled = {name: value / total for name, value in cleaned.items()}

    if not return_mapping:
        return reconciled
    return reconciled, reconciliation


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _estimate_carbon_count(name: str) -> int | None:
    match = re.search(r"C(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.match(r"^C(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _select_propane_sink(mechanism_species: set[str]) -> str:
    if "C3H8" in mechanism_species:
        return "C3H8"
    return _select_methane_sink(mechanism_species)


def _select_methane_sink(mechanism_species: set[str]) -> str:
    if "CH4" not in mechanism_species:
        raise ValueError("Mechanism does not contain CH4 for lumping missing species")
    return "CH4"
