import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .ga import initialize_population, fitness

@dataclass
class ABCOptions:
    food_count: int = 10
    limit: int = 5
    iterations: int = 20

def _neighbour(ind: np.ndarray) -> np.ndarray:
    new = ind.copy()
    idx = np.random.randint(len(ind))
    new[idx] = 1 - new[idx]
    return new

def run_abc(genome_length: int, eval_fn: Callable[[np.ndarray], float], options: ABCOptions = ABCOptions()) -> Tuple[np.ndarray, List[float]]:
    """Artificial Bee Colony algorithm."""
    foods = initialize_population(options.food_count, genome_length)
    fits = fitness(foods, eval_fn)
    trials = np.zeros(options.food_count, dtype=int)
    best_idx = fits.argmax()
    best_food = foods[best_idx].copy()
    best_fit = fits[best_idx]
    history: List[float] = [best_fit]
    for _ in range(options.iterations):
        # Employed bees
        for i in range(options.food_count):
            candidate = _neighbour(foods[i])
            cand_fit = eval_fn(candidate)
            if cand_fit > fits[i]:
                foods[i] = candidate
                fits[i] = cand_fit
                trials[i] = 0
            else:
                trials[i] += 1
        # Onlooker bees
        prob = fits / fits.sum()
        for _ in range(options.food_count):
            i = np.random.choice(options.food_count, p=prob)
            candidate = _neighbour(foods[i])
            cand_fit = eval_fn(candidate)
            if cand_fit > fits[i]:
                foods[i] = candidate
                fits[i] = cand_fit
                trials[i] = 0
            else:
                trials[i] += 1
        # Scout bees
        for i in range(options.food_count):
            if trials[i] >= options.limit:
                foods[i] = initialize_population(1, genome_length)[0]
                fits[i] = eval_fn(foods[i])
                trials[i] = 0
        idx = fits.argmax()
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best_food = foods[idx].copy()
        history.append(best_fit)
    return best_food, history
