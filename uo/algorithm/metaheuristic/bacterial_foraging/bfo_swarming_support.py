"""Social-interaction contract for BFO swarming."""

from __future__ import annotations

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
