"""
Exploration strategies for PBT


Explore functions takes in a set of hyperparameters and return slightly modified hyperparameters
"""

from collections.abc import Callable

import numpy as np


def multiplication_division(upper_bound: float = 1.1) -> Callable:
    def multiplication_division_func(hp_value: float) -> float:
        factor = round(np.random.uniform(1, upper_bound), 3)
        operator = np.random.choice([-1, 1])

        if operator == 1:
            return hp_value * factor
        return hp_value / factor

    return multiplication_division_func


def perturbation(factors: list[float] = [0.8, 1.25]) -> Callable:
    def perturbation_func(hp_value: float) -> float:
        factor = np.random.choice(factors)
        return hp_value * factor

    return perturbation_func


def perturbation_int_values(
    factors: list[float],
    min_value: int,
    max_value: int,
) -> Callable:
    def perturbation_func(hp_value: float) -> float:
        factor = np.random.choice(factors)
        value = int(hp_value * factor)
        return max(min_value, min(max_value, value))

    return perturbation_func
