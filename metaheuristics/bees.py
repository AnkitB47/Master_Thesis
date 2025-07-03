import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .ga import initialize_population, fitness

@dataclass
class BeesOptions:
    num_bees: int = 20
    elite_sites: int = 3
    selected_sites: int = 5
    elite_bees: int = 7
    other_bees: int = 3
    iterations: int = 20

def _neighbour(ind: np.ndarray) -> np.ndarray:
    new = ind.copy()
    idx = np.random.randint(len(ind))
    new[idx] = 1 - new[idx]
    return new

def run_bees(genome_length: int, eval_fn: Callable[[np.ndarray], float], options: BeesOptions = BeesOptions()) -> Tuple[np.ndarray, List[float]]:
    """Bees Algorithm implementation."""
    population = initialize_population(options.num_bees, genome_length)
    scores = fitness(population, eval_fn)
    best_idx = scores.argmax()
    best = population[best_idx].copy()
    best_score = scores[best_idx]
    history: List[float] = [best_score]
    for _ in range(options.iterations):
        order = np.argsort(-scores)
        new_pop = []
        # elite sites
        for i in order[:options.elite_sites]:
            site = population[i]
            neighbs = [site] + [_neighbour(site) for _ in range(options.elite_bees)]
            fits = fitness(np.array(neighbs), eval_fn)
            new_pop.append(neighbs[fits.argmax()])
        # selected sites
        for i in order[options.elite_sites:options.selected_sites]:
            site = population[i]
            neighbs = [site] + [_neighbour(site) for _ in range(options.other_bees)]
            fits = fitness(np.array(neighbs), eval_fn)
            new_pop.append(neighbs[fits.argmax()])
        # remaining bees random search
        remaining = options.num_bees - len(new_pop)
        if remaining > 0:
            new_pop.extend(initialize_population(remaining, genome_length))
        population = np.array(new_pop)
        scores = fitness(population, eval_fn)
        idx = scores.argmax()
        if scores[idx] > best_score:
            best_score = scores[idx]
            best = population[idx].copy()
        history.append(best_score)
    return best, history
