"""
The :mod:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_support` module describes the class :class:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_support.TabuSearchSupport`.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from abc import ABCMeta, abstractmethod
from typing import Generic, Hashable, Optional, TypeVar

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.single_solution_metaheuristic import SingleSolutionMetaheuristic
from uo.algorithm.metaheuristic.tabu_search.tabu_list import TabuList

R_co = TypeVar("R_co", covariant=True)
A_co = TypeVar("A_co", covariant=True)


class TabuSearchSupport(Generic[R_co, A_co], metaclass=ABCMeta):
    """
    This class represents placeholder for the neighborhood exploration used during execution of the
    Tabu Search metaheuristic. Concrete subclasses implement neighborhood generation and evaluation
    for a specific solution representation.
    """

    def __init__(self, dimension: int) -> None:
        """
        Create new `TabuSearchSupport` instance

        :param int dimension: dimension of the solution representation, used to determine neighborhood size
        """
        if dimension is None:
            raise ValueError('Parameter \'dimension\' must exists.')
        if not isinstance(dimension, int):
            raise TypeError('Parameter \'dimension\' must be int.')
        self.__dimension = dimension

    @abstractmethod
    def copy(self):
        """
        Copy the current object

        :return: new instance with the same properties
        :rtype: :class:`TabuSearchSupport`
        """
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """
        Property getter for the dimension of the solution representation

        :return: dimension of the solution representation
        :rtype: int
        """
        return self.__dimension

    @abstractmethod
    def best_neighbor_move(self, problem: Problem, solution: Solution[R_co, A_co], tabu_list: TabuList,
            optimizer: SingleSolutionMetaheuristic) -> Optional[Hashable]:
        """
        Explores the neighborhood of the supplied solution, and moves it (in place) to the best
        admissible neighbor -- the best neighbor that is not currently tabu, unless a non-tabu-listed
        neighbor is strictly better than the best solution found so far (aspiration criterion).

        :param `Problem` problem: problem that is solved
        :param `Solution` solution: solution used for the problem that is solved -- moved in place to
        the chosen neighbor, if any admissible neighbor exists
        :param `TabuList` tabu_list: structure that keeps track of currently forbidden moves
        :param `SingleSolutionMetaheuristic` optimizer: metaheuristic optimizer that is executed
        :return: move that was applied to reach the chosen neighbor, or `None` if no admissible
        neighbor could be found (e.g. execution should finish, or every neighbor is tabu)
        :rtype: `Optional[Hashable]`
        """
        raise NotImplementedError

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the `TabuSearchSupport` instance

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
        :return: string representation of the `TabuSearchSupport` instance
        :rtype: str
        """
        return group_start + 'dimension=' + str(self.dimension) + group_end

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
