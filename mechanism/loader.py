import cantera as ct
from typing import List, Dict

class Mechanism:
    """Wrapper around Cantera Solution for mechanism editing."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.solution = ct.Solution(file_path)

    @property
    def species_names(self) -> List[str]:
        return [s.name for s in self.solution.species()]

    def molecular_weights(self) -> Dict[str, float]:
        return {s.name: s.molecular_weight for s in self.solution.species()}

    def reactions(self) -> List[ct.Reaction]:
        return list(self.solution.reactions())

    def remove_species(self, species_list: List[str]):
        remaining_species = [s for s in self.solution.species() if s.name not in species_list]
        self.solution = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=remaining_species, reactions=self.solution.reactions())

    def remove_reactions(self, indexes: List[int]):
        reactions = [r for i, r in enumerate(self.solution.reactions()) if i not in indexes]
        self.solution = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=self.solution.species(), reactions=reactions)

    def save(self, out_path: str):
        ct.save(out_path, self.solution)
