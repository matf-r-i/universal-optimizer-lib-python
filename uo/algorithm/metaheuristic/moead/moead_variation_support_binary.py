"""
..  _py_moead_variation_support_binary:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_variation_support_binary`
module contains class
:class:`~uo.algorithm.metaheuristic.moead.moead_variation_support_binary.MoeadVariationSupportBinary`,
that represents variation support for the MOEA/D algorithm where solution
representation is binary.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(str(directory))
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
sys.path.append(str(directory.parent.parent.parent.parent))
sys.path.append(str(directory.parent.parent.parent.parent.parent))

from random import Random
from typing import TypeVar
from typing import Generic
from typing import Optional

import numpy as np

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.population_based_metaheuristic import PopulationBasedMetaheuristic
from uo.algorithm.metaheuristic.moead.moead_variation_support import MoeadVariationSupport

A_co = TypeVar("A_co", covariant=True)


class MoeadVariationSupportBinary(MoeadVariationSupport[np.ndarray, A_co]):

    def __init__(
            self,
            crossover_probability: float = 0.9,
            mutation_probability: Optional[float] = None
    ) -> None:
        """
        Create new binary MOEA/D variation support instance.
        """
        self.__crossover_probability: float = crossover_probability
        self.__mutation_probability: Optional[float] = mutation_probability

    def copy(self):
        """
        Copy the current instance.

        :return: copied instance
        :rtype: MoeadVariationSupportBinary
        """
        return MoeadVariationSupportBinary(
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability
        )

    @property
    def crossover_probability(self) -> float:
        """
        Property getter for crossover probability.
        """
        return self.__crossover_probability

    @property
    def mutation_probability(self) -> Optional[float]:
        """
        Property getter for mutation probability.
        """
        return self.__mutation_probability

    def generate_offspring(
            self,
            problem: Problem,
            parent1: Solution[np.ndarray, A_co],
            parent2: Solution[np.ndarray, A_co],
            child: Solution[np.ndarray, A_co],
            optimizer: PopulationBasedMetaheuristic
    ) -> None:
        """
        Generate one binary offspring using one-point crossover and bit-flip mutation.
        """
        if parent1.representation is None or parent2.representation is None:
            return

        x1 = np.asarray(parent1.representation, dtype=int)
        x2 = np.asarray(parent2.representation, dtype=int)

        rng = Random(optimizer.random_seed + optimizer.evaluation if optimizer.random_seed is not None else None)

        n = len(x1)
        p_m = self.mutation_probability
        if p_m is None:
            p_m = 1.0 / n

        if rng.random() < self.crossover_probability:
            point = rng.randint(1, n - 1)
            y = np.concatenate([x1[:point], x2[point:]])
        else:
            y = x1.copy()

        for i in range(n):
            if rng.random() < p_m:
                y[i] = 1 - y[i]

        child.representation = y
        optimizer.write_output_values_if_needed("before_evaluation", "b_e")
        optimizer.evaluation += 1
        child.evaluate(problem)
        optimizer.write_output_values_if_needed("after_evaluation", "b_e")

    def string_rep(
            self,
            delimiter: str,
            indentation: int = 0,
            indentation_symbol: str = '',
            group_start: str = '{',
            group_end: str = '}'
    ) -> str:
        """
        String representation of the support instance.
        """
        return 'MoeadVariationSupportBinary'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')