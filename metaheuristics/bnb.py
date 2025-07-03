import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class BnBOptions:
    max_size: int = 10


def run_bnb(genome_length: int, eval_fn: Callable[[np.ndarray], float], options: BnBOptions = BnBOptions()) -> Tuple[np.ndarray, List[float]]:
    """Branch and Bound exhaustive search for small genomes."""
    if genome_length > options.max_size:
        raise ValueError("Genome too large for branch-and-bound")
    best: np.ndarray = np.zeros(genome_length, dtype=int)
    best_score = -np.inf
    history: List[float] = []

    def dfs(prefix: List[int], idx: int):
        nonlocal best_score, best
        if idx == genome_length:
            ind = np.array(prefix)
            score = eval_fn(ind)
            history.append(score)
            if score > best_score:
                best_score = score
                best = ind.copy()
            return
        dfs(prefix + [0], idx + 1)
        dfs(prefix + [1], idx + 1)

    dfs([], 0)
    return best, history
