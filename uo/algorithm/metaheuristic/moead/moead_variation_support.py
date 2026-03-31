"""
..  _py_moead_variation_support:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_variation_support`
module describes the class
:class:`~uo.algorithm.metaheuristic.moead.moead_variation_support.MoeadVariationSupport`.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))

from abc import ABCMeta, abstractmethod
from typing import TypeVar
from typing import Generic

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.population_based_metaheuristic import PopulationBasedMetaheuristic

R_co = TypeVar("R_co", covariant=True)
A_co = TypeVar("A_co", covariant=True)


class MoeadVariationSupport(Generic[R_co, A_co], metaclass=ABCMeta):

    @abstractmethod
    def copy(self):
        """
        Copy the current object.

        :return: new instance with the same properties
        :rtype: :class:`MoeadVariationSupport`
        """
        raise NotImplementedError

    @abstractmethod
    def generate_offspring(
            self,
            problem: Problem,
            parent1: Solution[R_co, A_co],
            parent2: Solution[R_co, A_co],
            child: Solution[R_co, A_co],
            optimizer: PopulationBasedMetaheuristic
    ) -> None:
        """
        Generate one offspring from two parents.

        :param Problem problem: problem that is solved
        :param Solution[R_co,A_co] parent1: first parent
        :param Solution[R_co,A_co] parent2: second parent
        :param Solution[R_co,A_co] child: child solution that is created
        :param PopulationBasedMetaheuristic optimizer: optimizer that is executed
        :return: None
        """
        raise NotImplementedError