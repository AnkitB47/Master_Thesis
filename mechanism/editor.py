"""Utilities for editing CTI mechanism files."""

from __future__ import annotations

from typing import Iterable


def remove_species_cti(cti_path: str, species: Iterable[str], out_path: str) -> None:
    """Remove species from a CTI file and write to ``out_path``."""
    with open(cti_path) as f:
        lines = f.readlines()

    with open(out_path, "w") as f:
        for line in lines:
            if any(sp in line and line.strip().startswith("species") for sp in species):
                continue
            f.write(line)


def remove_reactions_cti(cti_path: str, indices: Iterable[int], out_path: str) -> None:
    """Remove reactions by index from a CTI file."""
    with open(cti_path) as f:
        lines = f.readlines()

    current = -1
    with open(out_path, "w") as f:
        for line in lines:
            if line.strip().startswith("reaction"):
                current += 1
            if current in indices:
                continue
            f.write(line)
