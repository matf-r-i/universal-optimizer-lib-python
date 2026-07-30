"""
The :mod:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support` module describes the classes
:class:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support.AcoEvaporationSupport`,
:class:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support.AcoEvaporationSupportFixed` and
:class:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support.AcoEvaporationSupportAdaptive`.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from abc import ABCMeta, abstractmethod

from uo.solution.solution import Solution


class AcoEvaporationSupport(metaclass=ABCMeta):
    """
    Placeholder for pheromone evaporation, deposit and clamping behavior,
    executed once per iteration of the ACO algorithm.
    """

    @abstractmethod
    def copy(self):
        """
        Copy the current object

        :return:  new instance with the same properties
        :rtype: :class:`AcoEvaporationSupport`
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, optimizer, iteration_best_solution: Solution, global_best_solution: Solution
    ) -> None:
        """
        Evaporate pheromone on all edges, deposit along the chosen tour,
        then clamp all values to `[tau_min, tau_max]`. Mutates
        `optimizer.tau` in place.

        :param optimizer: ACO optimizer executing the current iteration
        :param `Solution` iteration_best_solution: best solution found in the current iteration
        :param `Solution` global_best_solution: best solution found so far, across all iterations
        """
        raise NotImplementedError


class AcoEvaporationSupportFixed(AcoEvaporationSupport):
    """
    Deterministic parameter control (baseline): the pheromone deposit source
    switches from iteration-best to global-best at a fixed iteration count
    (50% of the maximum number of iterations); the evaporation rate never
    changes.
    """

    def copy(self) -> "AcoEvaporationSupportFixed":
        """
        Internal copy of the current instance.

        :return: new instance of class :class:`~uo.algorithm.metaheuristic.aco.AcoEvaporationSupportFixed`
        """
        return AcoEvaporationSupportFixed()

    def apply(
        self, optimizer, iteration_best_solution: Solution, global_best_solution: Solution
    ) -> None:
        """
        Evaporate at a constant rate; deposit source switches from
        iteration-best to global-best at a fixed iteration count.
        """
        n = optimizer.problem.n
        rho = optimizer.rho

        for a in range(n):
            for b in range(n):
                optimizer.tau[a][b] *= (1.0 - rho)

        max_iterations = optimizer.finish_control.iterations_max
        if max_iterations is not None and optimizer.iteration < max_iterations // 2:
            deposit_solution = iteration_best_solution
        else:
            deposit_solution = global_best_solution

        deposit_tour = deposit_solution.representation.tour
        cost = deposit_solution.tour_cost(optimizer.problem, deposit_tour)
        deposit = 1.0 / cost
        for k in range(n):
            a, b = deposit_tour[k], deposit_tour[(k + 1) % n]
            optimizer.tau[a][b] += deposit

        for a in range(n):
            for b in range(n):
                if optimizer.tau[a][b] > optimizer.tau_max:
                    optimizer.tau[a][b] = optimizer.tau_max
                elif optimizer.tau[a][b] < optimizer.tau_min:
                    optimizer.tau[a][b] = optimizer.tau_min


class AcoEvaporationSupportAdaptive(AcoEvaporationSupport):
    """
    Adaptive parameter control: the evaporation rate and the pheromone
    deposit source both react to the search's stagnation counter instead
    of a fixed iteration threshold.
    """

    def __init__(self, rho_scale: float = 2.0) -> None:
        """
        Create new instance of class
        :class:`~uo.algorithm.metaheuristic.aco.AcoEvaporationSupportAdaptive`.

        :param float rho_scale: how much the evaporation rate can grow at
            maximum stagnation, as a multiple of the base rate (e.g. 2.0
            means up to 3x the base rate)
        """
        self.__rho_scale = rho_scale

    def copy(self) -> "AcoEvaporationSupportAdaptive":
        """
        Internal copy of the current instance.

        :return: new instance of class :class:`~uo.algorithm.metaheuristic.aco.AcoEvaporationSupportAdaptive`
        """
        return AcoEvaporationSupportAdaptive(self.__rho_scale)

    def apply(
        self, optimizer, iteration_best_solution: Solution, global_best_solution: Solution
    ) -> None:
        """
        Evaporate at a rate that grows with the stagnation counter; deposit
        source switches to global-best once stagnation crosses 50% of the
        stagnation limit.
        """
        n = optimizer.problem.n
        stagnation_ratio = min(1.0, optimizer.stagnation_count / optimizer.stagnation_limit)
        rho = optimizer.rho * (1.0 + self.__rho_scale * stagnation_ratio)

        for a in range(n):
            for b in range(n):
                optimizer.tau[a][b] *= (1.0 - rho)

        deposit_solution = global_best_solution if stagnation_ratio > 0.5 else iteration_best_solution

        deposit_tour = deposit_solution.representation.tour
        cost = deposit_solution.tour_cost(optimizer.problem, deposit_tour)
        deposit = 1.0 / cost
        for k in range(n):
            a, b = deposit_tour[k], deposit_tour[(k + 1) % n]
            optimizer.tau[a][b] += deposit

        for a in range(n):
            for b in range(n):
                if optimizer.tau[a][b] > optimizer.tau_max:
                    optimizer.tau[a][b] = optimizer.tau_max
                elif optimizer.tau[a][b] < optimizer.tau_min:
                    optimizer.tau[a][b] = optimizer.tau_min
