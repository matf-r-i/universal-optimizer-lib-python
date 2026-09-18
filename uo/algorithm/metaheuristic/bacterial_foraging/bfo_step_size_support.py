"""Chemotactic step-size contract for BFO."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod


class BfoStepSizeSupport(metaclass=ABCMeta):
    """Define initialization and adaptation of bacterial step sizes."""

    @abstractmethod
    def copy(self) -> BfoStepSizeSupport:
        """Return an independent copy of this step-size strategy

        :return: copied step-size strategy
        :rtype: BfoStepSizeSupport
        """
        raise NotImplementedError

    @abstractmethod
    def initial_step_size(self) -> float:
        """Return the step assigned to a new or dispersed bacterium.

        :return: positive initial chemotactic step size
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def adapt(self, step_size: float, improved: bool) -> float:
        """Return the next step size after a movement attempt

        Fixed strategies may return ``step_size`` unchanged, while adaptive
        strategies may use ``improved`` to increase or decrease it

        :param float step_size: bacterium's current step size
        :param bool improved: whether the attempted move was accepted
        :return: step size for the next movement attempt
        :rtype: float
        """
        raise NotImplementedError


class BfoStepSizeSupportFixed(BfoStepSizeSupport):
    """Keep one positive chemotactic step size throughout execution"""

    def __init__(self, step_size: float) -> None:
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise TypeError("Parameter 'step_size' must be a number")
        if step_size <= 0:
            raise ValueError("Parameter 'step_size' must be positive")
        self.__step_size = float(step_size)

    @property
    def step_size(self) -> float:
        """Return the configured fixed chemotactic step size."""
        return self.__step_size

    def copy(self) -> BfoStepSizeSupportFixed:
        return BfoStepSizeSupportFixed(self.step_size)

    def initial_step_size(self) -> float:
        return self.step_size

    def adapt(self, step_size: float, improved: bool) -> float:
        return step_size


class BfoStepSizeSupportAdaptive(BfoStepSizeSupport):
    """Adapt each bacterium's step and clamp it to configured bounds"""

    def __init__(
        self,
        initial_step_size: float,
        minimum_step_size: float,
        maximum_step_size: float,
        increase_factor: float = 1.05,
        decrease_factor: float = 0.95,
    ) -> None:
        values = {
            "initial_step_size": initial_step_size,
            "minimum_step_size": minimum_step_size,
            "maximum_step_size": maximum_step_size,
            "increase_factor": increase_factor,
            "decrease_factor": decrease_factor,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Parameter '{name}' must be a number")
        if minimum_step_size <= 0 or maximum_step_size <= 0:
            raise ValueError("Step-size bounds must be positive")
        if minimum_step_size > maximum_step_size:
            raise ValueError("Minimum step size must not exceed maximum step size")
        if not minimum_step_size <= initial_step_size <= maximum_step_size:
            raise ValueError("Initial step size must be within the configured bounds")
        if increase_factor < 1:
            raise ValueError("Increase factor must be at least 1")
        if not 0 < decrease_factor <= 1:
            raise ValueError("Decrease factor must be in (0, 1]")
        self.__initial_step_size = float(initial_step_size)
        self.__minimum_step_size = float(minimum_step_size)
        self.__maximum_step_size = float(maximum_step_size)
        self.__increase_factor = float(increase_factor)
        self.__decrease_factor = float(decrease_factor)

    def copy(self) -> BfoStepSizeSupportAdaptive:
        return BfoStepSizeSupportAdaptive(
            self.initial_step_size_value,
            self.minimum_step_size,
            self.maximum_step_size,
            self.increase_factor,
            self.decrease_factor,
        )

    @property
    def initial_step_size_value(self) -> float:
        """Return the initial value used for adaptive step sizes."""
        return self.__initial_step_size

    @property
    def minimum_step_size(self) -> float:
        """Return the lower clamp for adaptive step sizes."""
        return self.__minimum_step_size

    @property
    def maximum_step_size(self) -> float:
        """Return the upper clamp for adaptive step sizes."""
        return self.__maximum_step_size

    @property
    def increase_factor(self) -> float:
        """Return the factor applied after an improving move."""
        return self.__increase_factor

    @property
    def decrease_factor(self) -> float:
        """Return the factor applied after a rejected move."""
        return self.__decrease_factor

    def initial_step_size(self) -> float:
        return self.initial_step_size_value

    def adapt(self, step_size: float, improved: bool) -> float:
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise TypeError("Parameter 'step_size' must be a number")
        if not isinstance(improved, bool):
            raise TypeError("Parameter 'improved' must be 'bool'")
        factor = self.increase_factor if improved else self.decrease_factor
        return min(
            self.maximum_step_size,
            max(self.minimum_step_size, step_size * factor),
        )
