"""Expose available metaheuristic optimisers.

Only the Genetic Algorithm (GA) is retained in the cleaned up codebase.
Other experimental algorithms have been removed.
"""

from .ga import run_ga, GAOptions

__all__ = ["run_ga", "GAOptions"]

