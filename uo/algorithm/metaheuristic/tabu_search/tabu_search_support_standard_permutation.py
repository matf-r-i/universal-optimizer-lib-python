"""
..  _py_tabu_search_support_standard_permutation:

The :mod:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_support_standard_permutation` contains class
:class:`~uo.algorithm.metaheuristic.tabu_search.TabuSearchSupportStandardPermutation`, that represents Tabu
Search neighborhood exploration support, where permutation (`list[int]`) representation of the problem has
been used.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from typing import Hashable, Optional, TypeVar

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.single_solution_metaheuristic import SingleSolutionMetaheuristic
from uo.algorithm.metaheuristic.tabu_search.tabu_list import TabuList
from uo.algorithm.metaheuristic.tabu_search.tabu_search_support import TabuSearchSupport

A_co = TypeVar("A_co", covariant=True)


class TabuSearchSupportStandardPermutation(TabuSearchSupport[list[int], A_co]):
    """
    Standard Tabu Search neighborhood exploration support for solutions with permutation (`list[int]`)
    representation. Neighborhood is defined by swapping two positions within the permutation -- for a
    permutation of length `n`, there are `n * (n - 1) / 2` such neighbors.
    """

    def __init__(self, dimension: int) -> None:
        """
        Create new `TabuSearchSupportStandardPermutation` instance
        """
        super().__init__(dimension=dimension)

    def copy(self) -> 'TabuSearchSupportStandardPermutation':
        """
        Copy the `TabuSearchSupportStandardPermutation`

        :return: new `TabuSearchSupportStandardPermutation` instance with the same properties
        :rtype: `TabuSearchSupportStandardPermutation`
        """
        return TabuSearchSupportStandardPermutation(self.dimension)

    def best_neighbor_move(self, problem: Problem, solution: Solution[list[int], A_co], tabu_list: TabuList,
            optimizer: SingleSolutionMetaheuristic) -> Optional[Hashable]:
        """
        Explores the swap-neighborhood of the supplied permutation solution, and moves it (in place) to
        the best admissible neighbor -- the best neighbor whose swap is not currently tabu, unless a
        tabu neighbor is strictly better than the best solution found so far (aspiration criterion).

        :param `Problem` problem: problem that is solved
        :param `Solution` solution: permutation solution used for the problem that is solved
        :param `TabuList` tabu_list: structure that keeps track of currently forbidden swaps
        :param `SingleSolutionMetaheuristic` optimizer: metaheuristic optimizer that is executed
        :return: `(i, j)` positions swapped to reach the chosen neighbor, or `None` if no admissible
        neighbor could be found
        :rtype: `Optional[tuple[int, int]]`
        """
        if optimizer.should_finish():
            return None
        representation: list[int] = solution.representation
        n: int = len(representation)
        if n < 2:
            return None
        start_sol: Solution = solution.copy()
        best_sol: Optional[Solution] = None
        best_move: Optional[tuple[int, int]] = None
        for i in range(n - 1):
            for j in range(i + 1, n):
                if optimizer.should_finish():
                    solution.copy_from(start_sol)
                    return None
                move: tuple[int, int] = (i, j)
                representation[i], representation[j] = representation[j], representation[i]
                optimizer.write_output_values_if_needed("before_evaluation", "b_e")
                optimizer.evaluation += 1
                solution.evaluate(problem)
                optimizer.write_output_values_if_needed("after_evaluation", "a_e")
                is_tabu: bool = tabu_list.contains(move)
                aspiration: Optional[bool] = solution.is_better(optimizer.best_solution, problem)
                if not is_tabu or aspiration:
                    if best_sol is None or solution.is_better(best_sol, problem):
                        best_sol = solution.copy()
                        best_move = move
                representation[i], representation[j] = representation[j], representation[i]
        if best_sol is None:
            solution.copy_from(start_sol)
            return None
        solution.copy_from(best_sol)
        return best_move

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the tabu search support instance

        :param delimiter: delimiter between fields
        :type delimiter: str
        :param indentation: level of indentation
        :type indentation: int, optional, default value 0
        :param indentation_symbol: indentation symbol
        :type indentation_symbol: str, optional, default value ''
        :param group_start: group start string
        :type group_start: str, optional, default value '{'
        :param group_end: group end string
        :type group_end: str, optional, default value '}'
        :return: string representation of tabu search support instance
        :rtype: str
        """
        return 'TabuSearchSupportStandardPermutation'

    def __str__(self) -> str:
        """
        String representation of the tabu search support instance

        :return: string representation of the tabu search support instance
        :rtype: str
        """
        return self.string_rep('|')

    def __repr__(self) -> str:
        """
        Representation of the tabu search support instance

        :return: string representation of the tabu search support instance
        :rtype: str
        """
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        """
        Formatted the tabu search support instance

        :param str spec: format specification
        :return: formatted tabu search support instance
        :rtype: str
        """
        return self.string_rep('|')
