import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Sequence

@dataclass
class GAOptions:
    population_size: int = 20
    generations: int = 10
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    # minimum number of active ("1") genes allowed in an individual
    min_species: int = 1


def initialize_population(
    pop_size: int, genome_length: int, seeds: Sequence[np.ndarray] | None = None
) -> np.ndarray:
    """Return an initial population optionally seeded with provided genomes."""
    pop = np.random.randint(0, 2, size=(pop_size, genome_length))
    if seeds is not None:
        n = min(len(seeds), pop_size)
        pop[:n] = np.array(seeds[:n])
    return pop


def fitness(
    population: np.ndarray, eval_fn: Callable[[np.ndarray], tuple | float]
) -> Tuple[np.ndarray, List[tuple]]:
    """Evaluate a population and collect per-individual details."""
    scores: List[float] = []
    details: List[tuple] = []
    for ind in population:
        out = eval_fn(ind)
        if isinstance(out, tuple):
            score, *info = out
            scores.append(float(score))
            details.append(tuple(info))
        else:
            scores.append(float(out))
            details.append(())
    return np.array(scores), details


def selection(population: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Select individuals using fitness-proportionate selection.

    When all scores are equal or negative, this reduces to uniform sampling
    which helps maintain diversity instead of collapsing the population.
    """

    scores = scores - scores.min()
    if scores.sum() <= 0:
        idx = np.random.choice(len(population), size=len(population))
    else:
        probs = scores / scores.sum()
        idx = np.random.choice(len(population), size=len(population), p=probs)
    return population[idx]


def crossover(population: np.ndarray, rate: float, mask: np.ndarray | None = None) -> np.ndarray:
    next_pop = population.copy()
    for i in range(0, len(population), 2):
        if np.random.rand() < rate and i + 1 < len(population):
            point = np.random.randint(1, population.shape[1])
            next_pop[i, point:], next_pop[i + 1, point:] = (
                population[i + 1, point:].copy(),
                population[i, point:].copy(),
            )
    if mask is not None:
        next_pop &= mask
    return next_pop


def mutate(population: np.ndarray, rate: float, mask: np.ndarray | None = None) -> np.ndarray:
    mutation = np.random.rand(*population.shape) < rate
    if mask is not None:
        mutation &= mask
    population[mutation] = 1 - population[mutation]
    if mask is not None:
        population &= mask
    return population


def _enforce_min(pop: np.ndarray, min_size: int) -> None:
    if min_size <= 0:
        return
    for ind in pop:
        ones = int(ind.sum())
        if ones < min_size:
            zeros = np.where(ind == 0)[0]
            if zeros.size:
                idx = np.random.choice(zeros, min_size - ones, replace=False)
                ind[idx] = 1


def run_ga(
    genome_length: int,
    eval_fn: Callable[[np.ndarray], float],
    options: GAOptions = GAOptions(),
    return_history: bool = False,
    initial_population: np.ndarray | None = None,
    return_debug: bool = False,
    fixed_indices: Sequence[int] | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray | Tuple[np.ndarray, List[float]] | Tuple[np.ndarray, List[float], List[List[tuple]]]:
    """Run a simple genetic algorithm.

    Parameters
    ----------
    genome_length:
        Length of the binary genome.
    eval_fn:
        Fitness evaluation function returning a scalar value.
    options:
        Control parameters for the GA.
    return_history:
        If ``True`` the function also returns a list of best scores per
        generation.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, List[float]]
        The best genome found and optionally the score history.
    """

    pop = (
        initial_population
        if initial_population is not None
        else initialize_population(options.population_size, genome_length)
    )
    if mask is not None:
        pop &= mask
    if fixed_indices is not None:
        pop[:, list(fixed_indices)] = 1
    _enforce_min(pop, options.min_species)
    best = pop[0]
    best_score = -np.inf
    history: List[float] = []
    debug: List[List[tuple]] = []
    for _ in range(options.generations):
        scores, details = fitness(pop, eval_fn)
        if scores.max() > best_score:
            best_score = float(scores.max())
            best = pop[scores.argmax()]
        history.append(best_score)
        if return_debug:
            debug.append(
                [
                    (
                        details[i][0] if len(details[i]) > 0 else None,
                        details[i][1] if len(details[i]) > 1 else None,
                        details[i][2] if len(details[i]) > 2 else None,
                        scores[i],
                        details[i][3] if len(details[i]) > 3 else "",
                        pop[i].copy(),
                    )
                    for i in range(len(pop))
                ]
            )
        pop = selection(pop, scores)
        pop = crossover(pop, options.crossover_rate, mask)
        pop = mutate(pop, options.mutation_rate, mask)
        if fixed_indices is not None:
            pop[:, list(fixed_indices)] = 1
        _enforce_min(pop, options.min_species)
    if return_history and return_debug:
        return best, history, debug
    if return_history:
        return best, history
    return best
