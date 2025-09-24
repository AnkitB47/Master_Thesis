import cantera as ct
from typing import Dict, Tuple, Callable


def methane_air_mole_fractions(phi: float, diluent: str = "N2") -> Dict[str, float]:
    """Return mole fractions for a methane/air mixture at equivalence ``phi``."""
    x = {
        "CH4": 1.0,
        "O2": 2.0 / phi,
    }
    x[diluent] = 3.76 * x["O2"]
    s = sum(x.values())
    return {k: v / s for k, v in x.items()}

def methane_oxygen_mole_fractions(phi: float) -> dict[str, float]:
    # CH4 + 2 O2 (stoich); for phi>1 rich, O2 reduced
    CH4 = 1.0
    O2_stoich = 2.0
    O2 = O2_stoich / max(phi, 1e-6)
    tot = CH4 + O2
    return {"CH4": CH4/tot, "O2": O2/tot}

def mole_to_mass_fractions(sol: ct.Solution, x: Dict[str, float]) -> Dict[str, float]:
    """Convert mole-fraction mapping ``x`` to a mass-fraction dict."""
    sol.TPX = 300.0, ct.one_atm, x
    return {sp: float(y) for sp, y in zip(sol.species_names, sol.Y) if y > 0}


# ---------------------------------------------------------------------------
# Homogeneous-reactor presets from Prüfert et al. (2013)
# ---------------------------------------------------------------------------

def _preset_from_phi(sol: ct.Solution, phi: float, T0: float = 1000.0, p0: float = ct.one_atm) -> Tuple[float, float, Dict[str, float]]:
    x = methane_air_mole_fractions(phi)
    Y0 = mole_to_mass_fractions(sol, x)
    return T0, p0, Y0


def preset_HR1(sol: ct.Solution) -> Tuple[float, float, Dict[str, float]]:
    """Lean methane/air mixture (φ≈0.5)."""
    return _preset_from_phi(sol, 0.5)


def preset_HR2(sol: ct.Solution) -> Tuple[float, float, Dict[str, float]]:
    """Stoichiometric methane/air mixture (φ≈1.0)."""
    return _preset_from_phi(sol, 1.0)


def preset_HR3(sol: ct.Solution) -> Tuple[float, float, Dict[str, float]]:
    """Rich methane/air mixture (φ≈1.5)."""
    return _preset_from_phi(sol, 1.5)


HR_PRESETS: Dict[str, Callable[[ct.Solution], Tuple[float, float, Dict[str, float]]]] = {
    "HR1": preset_HR1,
    "HR2": preset_HR2,
    "HR3": preset_HR3,
}
