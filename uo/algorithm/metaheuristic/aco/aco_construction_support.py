"""
The :mod:`~uo.algorithm.metaheuristic.aco.aco_construction_support` module describes the class
:class:`~uo.algorithm.metaheuristic.aco.aco_construction_support.AcoConstructionSupport`.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from abc import ABCMeta, abstractmethod

from uo.problem.problem import Problem
from uo.solution.solution import Solution


class AcoConstructionSupport(metaclass=ABCMeta):
    """
    Placeholder for additional methods, specific for ACO solution construction
    and local search, which depend on the precise problem and solution type.
    """

    @abstractmethod
    def copy(self):
        """
        Copy the current object

        :return:  new instance with the same properties
        :rtype: :class:`AcoConstructionSupport`
        """
        raise NotImplementedError

    @abstractmethod
    def construct(self, problem: Problem, solution: Solution, optimizer) -> None:
        """
        Build one ant's candidate solution in place, using the optimizer's
        pheromone (`tau`) and heuristic (`eta`) matrices.

        :param `Problem` problem: problem that is being solved
        :param `Solution` solution: solution instance to construct into
        :param optimizer: ACO optimizer executing the construction
        """
        raise NotImplementedError

    @abstractmethod
    def local_search(self, problem: Problem, solution: Solution, optimizer) -> None:
        """
        Refine a solution in place. Called only on the iteration-best ant.

        :param `Problem` problem: problem that is being solved
        :param `Solution` solution: solution instance to refine
        :param optimizer: ACO optimizer executing the local search
        """
        raise NotImplementedError
