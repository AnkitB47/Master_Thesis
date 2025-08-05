from metaheuristics.ga import run_ga, GAOptions
import numpy as np


def simple_eval(genome: np.ndarray) -> float:
    return genome.sum()


def test_ga_improves_fitness():
    best, history = run_ga(
        5,
        simple_eval,
        GAOptions(population_size=4, generations=5),
        return_history=True,
    )
    assert best.sum() <= 5
    assert history[-1] >= history[0]
