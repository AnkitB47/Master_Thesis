from metaheuristics.ga import run_ga, GAOptions
import numpy as np

def simple_eval(genome: np.ndarray) -> float:
    return genome.sum()

def test_ga_runs():
    best, hist = run_ga(5, simple_eval, GAOptions(population_size=4, generations=5))
    assert best.sum() <= 5
    assert len(hist) == 5
