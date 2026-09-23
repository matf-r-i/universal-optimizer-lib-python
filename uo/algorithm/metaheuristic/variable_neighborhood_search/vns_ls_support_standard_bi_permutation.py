"""
.. _py_vns_ls_support_standard_bi_permutation:

The :mod:`~uo.algorithm.metaheuristic.variable_neighborhood_search.vns_ls_support_standard_bi_permutation`
contains class
:class:`~uo.algorithm.metaheuristic.variable_neighborhood_search.VnsLocalSearchSupportStandardBestImprovementPermutation`,
that represents VNS local search support for solutions whose representation is a permutation,
possibly with repetition, kept within a `list[int]`.
"""

from typing import TypeVar

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.single_solution_metaheuristic import SingleSolutionMetaheuristic
from uo.algorithm.metaheuristic.variable_neighborhood_search.vns_ls_support import \
    VnsLocalSearchSupport

A_co = TypeVar("A_co", covariant=True)


class VnsLocalSearchSupportStandardBestImprovementPermutation(
        VnsLocalSearchSupport[list[int], A_co]):
    """
    VNS local search support for the permutation representation, best improvement variant.
    Algorithm uses explored neighborhood which consists of all valid neighboring 
    permutations. Parameter `k` is only used for range checks.

    """

    def __init__(self, dimension: int) -> None:
        """
        Create new `VnsLocalSearchSupportStandardBestImprovementPermutation` instance

        :param dimension: length of the permutation that is used as representation
        :type dimension: int
        """
        super().__init__(dimension=dimension)

    def copy(self) -> 'VnsLocalSearchSupportStandardBestImprovementPermutation':
        """
        Copy the `VnsLocalSearchSupportStandardBestImprovementPermutation`

        :return: new instance with the same properties
        :rtype: `VnsLocalSearchSupportStandardBestImprovementPermutation`
        """
        return VnsLocalSearchSupportStandardBestImprovementPermutation(self.dimension)

    def local_search(self, k: int, problem: Problem, solution: Solution,
                     optimizer: SingleSolutionMetaheuristic) -> bool:
        """
        Executes "best improvement" variant of the local search algorithm, over the neighborhood
        formed by all swaps of two positions holding different values.

        :param k: parameter of the VNS, used for the range check only
        :type k: int
        :param problem: problem that is solved
        :type problem: `Problem`
        :param solution: solution used for the problem that is solved, modified in place
        :type solution: `Solution`
        :param optimizer: metaheuristic optimizer that is executed
        :type optimizer: `SingleSolutionMetaheuristic`
        :return: if the local search is successful
        :rtype: bool
        """
        if optimizer.should_finish():
            return False
        if k < optimizer.k_min or k > optimizer.k_max:
            return False
        representation: list[int] = list(solution.representation)
        if len(representation) != self.dimension:
            raise ValueError(
                f"Length of the representation is {len(representation)}, "
                f"while local search support is created for dimension {self.dimension}."
            )
        start_sol: Solution = solution.copy()
        best_sol: Solution = solution.copy()
        better_sol_found: bool = False
        for i in range(self.dimension - 1):
            for j in range(i + 1, self.dimension):
                if representation[i] == representation[j]:
                    continue
                representation[i], representation[j] = representation[j], representation[i]
                solution.representation = list(representation)
                if optimizer.should_finish():
                    solution.copy_from(start_sol)
                    return False
                optimizer.write_output_values_if_needed("before_evaluation", "b_e")
                optimizer.evaluation += 1
                solution.evaluate(problem)
                optimizer.write_output_values_if_needed("after_evaluation", "a_e")
                if solution.is_better(best_sol, problem):
                    better_sol_found = True
                    best_sol.copy_from(solution)
                representation[i], representation[j] = representation[j], representation[i]
        if better_sol_found:
            solution.copy_from(best_sol)
            return True
        solution.copy_from(start_sol)
        return False

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
                   group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the vns support instance
        :return: string representation of vns support instance
        :rtype: str
        """
        return 'VnsLocalSearchSupportStandardBestImprovementPermutation'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
