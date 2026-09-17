"""Chemotactic step-size contract for BFO."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod


class BfoStepSizeSupport(metaclass=ABCMeta):
    """Define initialization and adaptation of bacterial step sizes."""

    @abstractmethod
    def copy(self) -> BfoStepSizeSupport:
        """Return an independent copy of this step-size strategy.

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
        """Return the next step size after a movement attempt.

        Fixed strategies may return ``step_size`` unchanged, while adaptive
        strategies may use ``improved`` to increase or decrease it.

        :param float step_size: bacterium's current step size
        :param bool improved: whether the attempted move was accepted
        :return: step size for the next movement attempt
        :rtype: float
        """
        raise NotImplementedError
