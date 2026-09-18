"""Representation-dependent movement contract for BFO."""

from __future__ import annotations

import math

from abc import ABCMeta, abstractmethod
from random import Random

from uo.problem.problem import Problem
from uo.solution.solution import Solution


class BfoMovementSupport(metaclass=ABCMeta):
    """Define how a bacterium tumbles and moves in its representation space."""

    @abstractmethod
    def copy(self) -> BfoMovementSupport:
        """Return an independent copy of this movement strategy

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
        """Generate one representation-specific tumble direction."""
        raise NotImplementedError

    @abstractmethod
    def move(
        self,
        solution: Solution,
        direction: object,
        step_size: float,
        problem: Problem,
    ) -> Solution:
        """Create a moved candidate without mutating ``solution``."""
        raise NotImplementedError


class BfoMovementSupportReal(BfoMovementSupport):
    """Move list or tuple based real representations inside bounds."""

    def __init__(
        self,
        lower_bounds: list[float] | tuple[float, ...],
        upper_bounds: list[float] | tuple[float, ...],
    ) -> None:
        self._validate_bounds(lower_bounds, upper_bounds)
        self.__lower_bounds = tuple(float(value) for value in lower_bounds)
        self.__upper_bounds = tuple(float(value) for value in upper_bounds)

    @staticmethod
    def _validate_bounds(lower_bounds: object, upper_bounds: object) -> None:
        if not isinstance(lower_bounds, (list, tuple)):
            raise TypeError("Parameter 'lower_bounds' must be 'list' or 'tuple'")
        if not isinstance(upper_bounds, (list, tuple)):
            raise TypeError("Parameter 'upper_bounds' must be 'list' or 'tuple'")
        if not lower_bounds or len(lower_bounds) != len(upper_bounds):
            raise ValueError("Movement bounds must be non-empty and equally sized")
        for lower, upper in zip(lower_bounds, upper_bounds):
            if isinstance(lower, bool) or not isinstance(lower, (int, float)):
                raise TypeError("Movement bounds must contain numbers")
            if isinstance(upper, bool) or not isinstance(upper, (int, float)):
                raise TypeError("Movement bounds must contain numbers")
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("Movement bounds must be finite")
            if lower > upper:
                raise ValueError("Every lower bound must not exceed its upper bound")

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        """Return the inclusive lower bound for each real dimension."""
        return self.__lower_bounds

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        """Return the inclusive upper bound for each real dimension."""
        return self.__upper_bounds

    def copy(self) -> BfoMovementSupportReal:
        return BfoMovementSupportReal(self.lower_bounds, self.upper_bounds)

    def tumble_direction(
        self,
        solution: Solution,
        random_generator: Random,
    ) -> tuple[float, ...]:
        representation = solution.representation
        if not isinstance(representation, (list, tuple)):
            raise TypeError("Real BFO movement requires a list or tuple representation")
        if len(representation) != len(self.lower_bounds):
            raise ValueError("Solution representation dimension does not match movement bounds")
        while True:
            direction = tuple(random_generator.uniform(-1.0, 1.0) for _ in representation)
            norm = math.sqrt(sum(value * value for value in direction))
            if norm > 0.0:
                return tuple(value / norm for value in direction)

    def move(
        self,
        solution: Solution,
        direction: object,
        step_size: float,
        problem: Problem,
    ) -> Solution:
        representation = solution.representation
        if not isinstance(representation, (list, tuple)):
            raise TypeError("Real BFO movement requires a list or tuple representation")
        if not isinstance(direction, (list, tuple)) or len(direction) != len(representation):
            raise ValueError("Movement direction dimension does not match solution")
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise TypeError("Parameter 'step_size' must be a number")
        moved = []
        for value, delta, lower, upper in zip(
            representation, direction, self.lower_bounds, self.upper_bounds
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Real solution representations must contain numbers")
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                raise TypeError("Movement directions must contain numbers")
            moved.append(min(upper, max(lower, value + step_size * delta)))
        candidate = solution.copy()
        candidate.init_from(tuple(moved) if isinstance(representation, tuple) else moved, problem)
        return candidate
