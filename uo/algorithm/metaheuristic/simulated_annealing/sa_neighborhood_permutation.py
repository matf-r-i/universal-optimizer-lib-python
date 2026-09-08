"""
The :mod:`~uo.algorithm.metaheuristic.simulated_annealing.sa_neighborhood_permutation` contains
class :class:`~uo.algorithm.metaheuristic.simulated_annealing.sa_neighborhood_permutation.SaNeighborhoodPermutation`,
that represents Simulated Annealing neighborhood support for solutions whose representation is a
permutation, possibly with repetition, kept within a `list[int]`.
"""


from copy import deepcopy
import random

from uo.algorithm.metaheuristic.simulated_annealing.sa_neighborhood import SaNeighborhood

from uo.solution.solution import Solution
from uo.problem.problem import Problem

class SaNeighborhoodPermutation(SaNeighborhood):
    def __init__(self, dimension: int, k: int = 1) -> None:
        """
        Create new `SaNeighboorhoodPermutation` instance

        :param dimension: length of the permutation used as representation 
        :type dimension: int
        :param k: number of swaps that form a single move, defaults to 1
        :type k: int, optional
        """

        if not isinstance(dimension, int):
            raise TypeError("Parameter 'dimension' must be 'int'.")
        if dimension < 2:
            raise ValueError("Parameter 'dimension' must be at least 2.")
        if not isinstance(k, int):
            raise TypeError("Parameter 'k' must be 'int'.")
        if k < 1:
            raise ValueError("Parameter 'k' must be greater than zero.")
        self.dimension: int = dimension
        self.k: int = k

    def __copy__(self):
        return deepcopy(self)

    def copy(self) -> 'SaNeighborhoodPermutation':
        """Copy the current object

        :return: new instance with the same properties 
        :rtype: SaNeighborhoodPermutation
        """


        return self.__copy__()
    
    def generate_neighbor(self, solution: Solution, problem: Problem, optimizer=None) -> Solution:
        """Generate a neighbor by swapping `k` pairs of positions.

        :param solution:  solution whoose neighbor is generated
        :type solution: Solution
        :param problem: problem which is solved 
        :type problem: Problem
        :param optimizer: metaheuristic optimizer used for evaluation
        :return:  neighboring solution, evaluated
        :rtype: Solution
        """

        neighbor: Solution = solution.copy()
        representation: list[int] = list(neighbor.representation)
        if len(representation) != self.dimension:
            raise ValueError(
                f"Length of the representation is {len(representation)}, "
                f"while neighborhood is created for dimension {self.dimension}."
            )

        limit: int = 10000
        swaps_done: int = 0
        tries: int = 0

        while swaps_done < self.k and tries < limit:
            tries += 1
            i: int = random.randrange(len(representation))
            j: int = random.randrange(len(representation))
            if representation[i] == representation[j]:
                continue
            representation[i], representation[j] = representation[j], representation[i]
            swaps_done += 1
        if swaps_done == 0:
            return solution.copy()
        neighbor.representation = representation
        if optimizer is not None and hasattr(optimizer, "write_output_values_if_needed"):
            optimizer.write_output_values_if_needed("before_evaluation", "b_e")
        neighbor.evaluate(problem)
        if optimizer is not None:
            if hasattr(optimizer, "evaluation"):
                optimizer.evaluation += 1
            if hasattr(optimizer, "write_output_values_if_needed"):
                optimizer.write_output_values_if_needed("after_evaluation", "a_e")
        return neighbor

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
                   group_start: str = '{', group_end: str = '}') -> str:
        """String representation of the SA neighborhood instance

        :return: string representation of the instance 
        :rtype: str
        """
        return f'SaNeighborhoodPermutation{group_start}dimension={self.dimension}' \
               f'{delimiter}k={self.k}{group_end}'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')