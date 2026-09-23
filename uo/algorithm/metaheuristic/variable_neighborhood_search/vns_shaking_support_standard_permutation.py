"""
.. _py_vns_shaking_support_standard_permutation:

The :mod:`~uo.algorithm.metaheuristic.variable_neighborhood_search.vns_shaking_support_standard_permutation`
contains class
:class:`~uo.algorithm.metaheuristic.variable_neighborhood_search.VnsShakingSupportStandardPermutation`,
that represents VNS shaking support for solutions whose representation is a permutation, possibly
with repetition, kept within a `list[int]`.
"""


import random
from typing import TypeVar

from uo.algorithm.metaheuristic.variable_neighborhood_search.vns_shaking_support import VnsShakingSupport
from uo.algorithm.metaheuristic.single_solution_metaheuristic import SingleSolutionMetaheuristic


from uo.solution.solution import Solution
from uo.problem.problem import Problem

A_co = TypeVar("A_co",covariant=True)

class VnsShakingSupportStandardPermutation(VnsShakingSupport[list[int],A_co]):
    def __init__(self, dimension: int) -> None:
        """Create new `VnsShakingSupportStandardPermutation` instance

        :param dimension:  length of the permutation that is used as a represenattion
        :type dimension: int
        """
        super().__init__(dimension=dimension)

    def copy(self) -> 'VnsShakingSupportStandardPermutation':
        """Copy the current object

        :return: new instance with the same properties 
        :rtype: VnsShakingSupportStandardPermutation
        """


        return VnsShakingSupportStandardPermutation(self.dimension)
    
    def shaking(self, k: int, problem: Problem, solution: Solution, 
                optimizer: SingleSolutionMetaheuristic) -> bool:



        if optimizer.should_finish():
            return False
        if k < optimizer.k_min or k > optimizer.k_max:
            return False

        
        representation: list[int] = list(solution.representation)
        if len(representation) != self.dimension:
            raise ValueError(
                f"Length of the representation is {len(representation)}, "
                f"while neighborhood is created for dimension {self.dimension}."
            )

        limit: int = 10000
        swaps_done: int = 0
        tries: int = 0

        while swaps_done < k and tries < limit:
            tries += 1
            i: int = random.randrange(len(representation))
            j: int = random.randrange(len(representation))
            if representation[i] == representation[j]:
                continue
            representation[i], representation[j] = representation[j], representation[i]
            swaps_done += 1
        if swaps_done < k:
            return False


        solution.representation = representation
        optimizer.write_output_values_if_needed("before_evaluation","b_e")
        optimizer.evaluation += 1
        solution.evaluate(problem)
        optimizer.write_output_values_if_needed("after_evaluation","a_e")
        return True
    
    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
                   group_start: str = '{', group_end: str = '}') -> str:
        """String representation of the VNS support instance

        :return: string representation of the instance 
        :rtype: str
        """

        return 'VnsShakingSupportStandardPermutation'


    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')