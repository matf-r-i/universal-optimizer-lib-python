"""
..  _py_moead_variation_support_real:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_variation_support_real`
module contains class
:class:`~uo.algorithm.metaheuristic.moead.moead_variation_support_real.MoeadVariationSupportReal`,
that represents variation support for the MOEA/D algorithm where solution
representation is real-valued.
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


class MoeadVariationSupportReal(MoeadVariationSupport[np.ndarray, A_co]):

    def __init__(
            self,
            crossover_probability: float = 1.0,
            mutation_probability: Optional[float] = None,
            eta_c: float = 20.0,
            eta_m: float = 20.0
    ) -> None:
        """
        Create new real-valued MOEA/D variation support instance.
        """
        self.__crossover_probability: float = crossover_probability
        self.__mutation_probability: Optional[float] = mutation_probability
        self.__eta_c: float = eta_c
        self.__eta_m: float = eta_m

    def copy(self):
        """
        Copy the current instance.

        :return: copied instance
        :rtype: MoeadVariationSupportReal
        """
        return MoeadVariationSupportReal(
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            eta_c=self.eta_c,
            eta_m=self.eta_m
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

    @property
    def eta_c(self) -> float:
        """
        Property getter for SBX distribution index.
        """
        return self.__eta_c

    @property
    def eta_m(self) -> float:
        """
        Property getter for polynomial mutation distribution index.
        """
        return self.__eta_m

    def _get_bounds(self, problem: Problem, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Obtain lower and upper bounds from problem object.

        Expected attributes are ``lower_bound`` and ``upper_bound``.
        """
        if not hasattr(problem, 'lower_bound'):
            raise AttributeError("Problem must define attribute 'lower_bound' for real-valued MOEA/D.")
        if not hasattr(problem, 'upper_bound'):
            raise AttributeError("Problem must define attribute 'upper_bound' for real-valued MOEA/D.")

        xl = np.asarray(problem.lower_bound, dtype=float)
        xu = np.asarray(problem.upper_bound, dtype=float)

        if xl.shape[0] != n or xu.shape[0] != n:
            raise ValueError("Bounds size must match solution representation length.")

        return xl, xu

    def _sbx_crossover(
            self,
            rng: Random,
            p1: np.ndarray,
            p2: np.ndarray,
            xl: np.ndarray,
            xu: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute SBX crossover.
        """
        n = p1.shape[0]

        if rng.random() > self.crossover_probability:
            return p1.copy(), p2.copy()

        c1 = p1.copy()
        c2 = p2.copy()

        for i in range(n):
            if rng.random() <= 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1 = min(p1[i], p2[i])
                    x2 = max(p1[i], p2[i])
                    u = rng.random()

                    beta = 1.0 + (2.0 * (x1 - xl[i]) / (x2 - x1))
                    alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0))
                    child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu[i] - x2) / (x2 - x1))
                    alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0))
                    child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                    child1 = np.clip(child1, xl[i], xu[i])
                    child2 = np.clip(child2, xl[i], xu[i])

                    if rng.random() <= 0.5:
                        c1[i] = child2
                        c2[i] = child1
                    else:
                        c1[i] = child1
                        c2[i] = child2
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]

        return c1, c2

    def _polynomial_mutation(
            self,
            rng: Random,
            x: np.ndarray,
            xl: np.ndarray,
            xu: np.ndarray
    ) -> np.ndarray:
        """
        Execute polynomial mutation.
        """
        n = x.shape[0]
        p_m = self.mutation_probability
        if p_m is None:
            p_m = 1.0 / n

        y = x.copy()
        for i in range(n):
            if rng.random() <= p_m:
                if xu[i] - xl[i] <= 0.0:
                    continue

                delta1 = (y[i] - xl[i]) / (xu[i] - xl[i])
                delta2 = (xu[i] - y[i]) / (xu[i] - xl[i])
                u = rng.random()
                mut_pow = 1.0 / (self.eta_m + 1.0)

                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (self.eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (self.eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                y[i] = y[i] + deltaq * (xu[i] - xl[i])
                y[i] = np.clip(y[i], xl[i], xu[i])

        return y

    def generate_offspring(
            self,
            problem: Problem,
            parent1: Solution[np.ndarray, A_co],
            parent2: Solution[np.ndarray, A_co],
            child: Solution[np.ndarray, A_co],
            optimizer: PopulationBasedMetaheuristic
    ) -> None:
        """
        Generate one real-valued offspring using SBX and polynomial mutation.
        """
        if parent1.representation is None or parent2.representation is None:
            return

        p1 = np.asarray(parent1.representation, dtype=float)
        p2 = np.asarray(parent2.representation, dtype=float)

        rng = Random(optimizer.random_seed + optimizer.evaluation if optimizer.random_seed is not None else None)

        xl, xu = self._get_bounds(problem, len(p1))

        c1, c2 = self._sbx_crossover(rng, p1, p2, xl, xu)
        y = c1 if rng.random() < 0.5 else c2
        y = self._polynomial_mutation(rng, y, xl, xu)

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
        return 'MoeadVariationSupportReal'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')