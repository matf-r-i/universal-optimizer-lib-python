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

    def __init__(self, dimension: int) -> None:
        if dimension is None:
            raise ValueError('Parameter \'dimension\' must exists.')
        if not isinstance(dimension, int):
            raise TypeError('Parameter \'dimension\' must be int.')
        self.__dimension = dimension

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return self.__dimension

    @abstractmethod
    def best_neighbor_move(self, problem: Problem, solution: Solution[R_co, A_co], tabu_list: TabuList,
            optimizer: SingleSolutionMetaheuristic) -> Optional[Hashable]:
        raise NotImplementedError

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
        return group_start + 'dimension=' + str(self.dimension) + group_end

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
