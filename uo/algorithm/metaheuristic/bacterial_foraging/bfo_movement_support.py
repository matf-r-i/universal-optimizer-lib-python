"""Representation-dependent movement contract for BFO."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from random import Random

from uo.problem.problem import Problem
from uo.solution.solution import Solution


class BfoMovementSupport(metaclass=ABCMeta):
    """Define how a bacterium tumbles and moves in its representation space."""

    @abstractmethod
    def copy(self) -> BfoMovementSupport:
        """Return an independent copy of this movement strategy.

        :return: copied movement strategy
        :rtype: BfoMovementSupport
        """
        raise NotImplementedError

    @abstractmethod
    def tumble_direction(
        self,
        solution: Solution,
        random_generator: Random,
    ) -> object:
        """Generate one tumble direction for ``solution``.

        The returned object's concrete type is representation-specific and is
        passed back unchanged to :meth:`move` during a swim.

        :param Solution solution: bacterium whose direction is generated
        :param Random random_generator: optimizer-owned random generator
        :return: representation-specific direction
        :rtype: object
        """
        raise NotImplementedError

    @abstractmethod
    def move(
        self,
        solution: Solution,
        direction: object,
        step_size: float,
        problem: Problem,
    ) -> Solution:
        """Create a moved candidate without mutating ``solution``.

        Concrete strategies are responsible for representation handling and
        domain repair, such as clipping a real vector to its bounds.

        :param Solution solution: bacterium to move
        :param object direction: direction returned by :meth:`tumble_direction`
        :param float step_size: chemotactic step size
        :param Problem problem: problem being optimized
        :return: unevaluated moved candidate
        :rtype: Solution
        """
        raise NotImplementedError
