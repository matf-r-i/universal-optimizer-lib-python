"""Social-interaction contract for BFO swarming."""

from __future__ import annotations

import math

from abc import ABCMeta, abstractmethod

from uo.solution.solution import Solution


class BfoSwarmingSupport(metaclass=ABCMeta):
    """Calculate the temporary social contribution to bacterial fitness."""

    @abstractmethod
    def copy(self) -> BfoSwarmingSupport:
        """Return an independent copy of this swarming strategy.

        :return: copied swarming strategy
        :rtype: BfoSwarmingSupport
        """
        raise NotImplementedError

    @abstractmethod
    def interaction_value(
        self,
        bacterium: Solution,
        population: list[Solution],
    ) -> float:
        """Return the social-fitness adjustment for one bacterium.

        The optimizer adds this value to the bacterium's fitness. Therefore,
        positive values improve effective fitness under the library's
        higher-fitness-is-better convention. Implementations must not mutate
        the bacterium or overwrite its objective and fitness values.

        :param Solution bacterium: bacterium for which interaction is measured
        :param list[Solution] population: current bacterial population
        :return: social-fitness adjustment
        :rtype: float
        """
        raise NotImplementedError


class BfoSwarmingSupportIdle(BfoSwarmingSupport):
    """Disable social interaction between bacteria."""

    def copy(self) -> BfoSwarmingSupportIdle:
        return BfoSwarmingSupportIdle()

    def interaction_value(self, bacterium: Solution, population: list[Solution]) -> float:
        return 0.0


class BfoSwarmingSupportReal(BfoSwarmingSupport):
    """Calculate temporary social fitness from representation distances"""

    def __init__(
        self,
        attractant_depth: float,
        attractant_width: float,
        repellent_height: float,
        repellent_width: float,
    ) -> None:
        values = {
            "attractant_depth": attractant_depth,
            "attractant_width": attractant_width,
            "repellent_height": repellent_height,
            "repellent_width": repellent_width,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Parameter '{name}' must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Parameter '{name}' must be finite and non-negative")
        if attractant_width == 0 or repellent_width == 0:
            raise ValueError("Swarming widths must be positive")
        self.__attractant_depth = float(attractant_depth)
        self.__attractant_width = float(attractant_width)
        self.__repellent_height = float(repellent_height)
        self.__repellent_width = float(repellent_width)

    def copy(self) -> BfoSwarmingSupportReal:
        return BfoSwarmingSupportReal(
            self.attractant_depth,
            self.attractant_width,
            self.repellent_height,
            self.repellent_width,
        )

    @property
    def attractant_depth(self) -> float:
        return self.__attractant_depth

    @property
    def attractant_width(self) -> float:
        return self.__attractant_width

    @property
    def repellent_height(self) -> float:
        return self.__repellent_height

    @property
    def repellent_width(self) -> float:
        return self.__repellent_width

    def interaction_value(self, bacterium: Solution, population: list[Solution]) -> float:
        value = 0.0
        for other in population:
            distance = bacterium.representation_distance(
                bacterium.representation,
                other.representation,
            )
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                raise TypeError("Solution distance must be a number")
            squared_distance = float(distance) ** 2
            value += self.attractant_depth * math.exp(
                -self.attractant_width * squared_distance
            )
            value -= self.repellent_height * math.exp(
                -self.repellent_width * squared_distance
            )
        return value
