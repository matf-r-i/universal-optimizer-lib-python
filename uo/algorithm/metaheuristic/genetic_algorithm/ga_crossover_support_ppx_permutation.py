"""
..  _py_ga_crossover_support_ppx_permutation:

The :mod:`~uo.algorithm.metaheuristic.genetic_algorithm.ga_crossover_support_ppx_permutation` contains
class :class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaCrossoverSupportPpxPermutation`, that
represents GA crossover support for solutions whose representation is a permutation, possibly with
repetition, kept within a `list[int]`.
"""

from random import random, randrange
from typing import TypeVar

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.population_based_metaheuristic import PopulationBasedMetaheuristic
from uo.algorithm.metaheuristic.genetic_algorithm.ga_crossover_support import GaCrossoverSupport

A_co = TypeVar("A_co", covariant=True)


class GaCrossoverSupportPpxPermutation(GaCrossoverSupport[list[int], A_co]):
    """
    GA crossover support for the permutation representation, implementing the Precedence
    Preservative Crossover (PPX) of Bierwirth and Mattfeld.

    """

    def __init__(self, crossover_probability: float) -> None:
        """
        Create new `GaCrossoverSupportPpxPermutation` instance

        :param crossover_probability: probability that two parents are actually crossed
        :type crossover_probability: float
        """
        if not isinstance(crossover_probability, (float, int)):
            raise TypeError("Parameter 'crossover_probability' must be 'float'.")
        if crossover_probability < 0 or crossover_probability > 1:
            raise ValueError("Parameter 'crossover_probability' must be between 0 and 1.")
        self.__crossover_probability: float = float(crossover_probability)

    def copy(self) -> 'GaCrossoverSupportPpxPermutation':
        """
        Copy the `GaCrossoverSupportPpxPermutation` instance

        :return: new `GaCrossoverSupportPpxPermutation` instance with the same properties
        :rtype: `GaCrossoverSupportPpxPermutation`
        """
        return GaCrossoverSupportPpxPermutation(self.crossover_probability)

    @property
    def crossover_probability(self) -> float:
        """
        Getter for crossover probability

        :return: crossover probability
        :rtype: float
        """
        return self.__crossover_probability

    @staticmethod
    def offspring(parent1: list[int], parent2: list[int]) -> list[int]:
        """
        Creates a single offspring out of two permutations, according to PPX.

        :param parent1: permutation of the first parent
        :type parent1: list[int]
        :param parent2: permutation of the second parent
        :type parent2: list[int]
        :return: permutation of the offspring
        :rtype: list[int]
        """
        remaining1: list[int] = list(parent1)
        remaining2: list[int] = list(parent2)
        child: list[int] = []
        for _ in range(len(parent1)):
            value: int = remaining1[0] if randrange(2) else remaining2[0]
            child.append(value)
            remaining1.remove(value)
            remaining2.remove(value)
        return child

    def crossover(self, problem: Problem, solution1: Solution, solution2: Solution,
                  child1: Solution, child2: Solution,
                  optimizer: PopulationBasedMetaheuristic) -> None:
        """
        Executes crossover within GA, by applying PPX twice, with an independent sequence of
        random draws for each of the two children.

        :param problem: problem that is solved
        :type problem: `Problem`
        :param solution1: first parent
        :type solution1: `Solution`
        :param solution2: second parent
        :type solution2: `Solution`
        :param child1: first child
        :type child1: `Solution`
        :param child2: second child
        :type child2: `Solution`
        :param optimizer: optimizer that is executed
        :type optimizer: `PopulationBasedMetaheuristic`
        :rtype: None
        """
        if solution1.representation is None or solution2.representation is None:
            child1.copy_from(solution1)
            child2.copy_from(solution2)
            return
        if len(solution1.representation) != len(solution2.representation):
            raise ValueError(
                f"Parents are of different length, {len(solution1.representation)} "
                f"and {len(solution2.representation)}."
            )
        if sorted(solution1.representation) != sorted(solution2.representation):
            raise ValueError("Parents are not permutations of the same multiset of values.")
        if random() > self.crossover_probability:
            child1.copy_from(solution1)
            child2.copy_from(solution2)
            return
        child1.representation = self.offspring(solution1.representation, solution2.representation)
        child2.representation = self.offspring(solution1.representation, solution2.representation)
        optimizer.write_output_values_if_needed("before_evaluation", "b_e")
        optimizer.evaluation += 2
        child1.evaluate(problem)
        child2.evaluate(problem)
        optimizer.write_output_values_if_needed("after_evaluation", "a_e")

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
                   group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the ga support structure

        :return: string representation of ga support instance
        :rtype: str
        """
        return f'GaCrossoverSupportPpxPermutation{group_start}' \
               f'crossover_probability={self.crossover_probability}{group_end}'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
