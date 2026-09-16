"""
..  _py_ga_mutation_support_swap_permutation:

The :mod:`~uo.algorithm.metaheuristic.genetic_algorithm.ga_mutation_support_swap_permutation` contains
class :class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaMutationSupportSwapPermutation`, that
represents GA mutation support for solutions whose representation is a permutation, possibly with
repetition, kept within a `list[int]`.
"""

from random import random, randrange
from typing import TypeVar

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.population_based_metaheuristic import PopulationBasedMetaheuristic
from uo.algorithm.metaheuristic.genetic_algorithm.ga_mutation_support import GaMutationSupport

A_co = TypeVar("A_co", covariant=True)


class GaMutationSupportSwapPermutation(GaMutationSupport[list[int], A_co]):
    """
    GA mutation support for the permutation representation.

    """

    def __init__(self, mutation_probability: float) -> None:
        """
        Create new `GaMutationSupportSwapPermutation` instance

        :param mutation_probability: probability that a single position will be swapped 
        :type mutation_probability: float
        """
        if not isinstance(mutation_probability, (float, int)):
            raise TypeError("Parameter 'mutation_probability' must be 'float'.")
        if mutation_probability < 0 or mutation_probability > 1:
            raise ValueError("Parameter 'mutation_probability' must be between 0 and 1.")
        self.__mutation_probability: float = float(mutation_probability)

    def copy(self) -> 'GaMutationSupportSwapPermutation':
        """
        Copy the `GaMutationSupportSwapPermutation` instance

        :return: new `GaMutationSupportSwapPermutation` instance with the same properties
        :rtype: `GaMutationSupportSwapPermutation`
        """
        return GaMutationSupportSwapPermutation(self.mutation_probability)

    @property
    def mutation_probability(self) -> float:
        """
        Getter for mutation probability

        :return: mutation probability
        :rtype: float
        """
        return self.__mutation_probability

    def mutation(self, problem: Problem, solution: Solution,
                 optimizer: PopulationBasedMetaheuristic) -> None:
        """
        Executes mutation within GA, by swapping positions of the permutation.

        :param problem: problem that is solved
        :type problem: `Problem`
        :param solution: individual that is mutated, modified in place
        :type solution: `Solution`
        :param optimizer: optimizer that is executed
        :type optimizer: `PopulationBasedMetaheuristic`
        :rtype: None
        """
        if solution.representation is None:
            return
        representation: list[int] = list(solution.representation)
        length: int = len(representation)
        for i in range(length):
            if random() >= self.mutation_probability:
                continue
            j: int = randrange(length)
            if representation[i] == representation[j]:
                continue
            representation[i], representation[j] = representation[j], representation[i]
        solution.representation = representation
        optimizer.write_output_values_if_needed("before_evaluation", "b_e")
        optimizer.evaluation += 1
        solution.evaluate(problem)
        optimizer.write_output_values_if_needed("after_evaluation", "a_e")

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
                   group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the ga support structure

        :return: string representation of ga support instance
        :rtype: str
        """
        return f'GaMutationSupportSwapPermutation{group_start}' \
               f'mutation_probability={self.mutation_probability}{group_end}'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
