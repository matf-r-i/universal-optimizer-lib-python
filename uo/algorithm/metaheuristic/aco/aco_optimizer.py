"""
..  _py_aco_optimizer:
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from uo.problem.problem import Problem
from uo.solution.solution import Solution

from uo.algorithm.output_control import OutputControl
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.additional_statistics_control import AdditionalStatisticsControl
from uo.algorithm.metaheuristic.population_based_metaheuristic import PopulationBasedMetaheuristic


@dataclass
class AcoOptimizerConstructionParameters:
    """
    Instance of the class :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizerConstructionParameters`
    represents constructor parameters for the ACO algorithm.
    """
    construction_support: "AcoConstructionSupport" = None
    evaporation_support: "AcoEvaporationSupport" = None
    n_ants: Optional[int] = None
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.02
    p_best: float = 0.05
    stagnation_limit: int = 100
    finish_control: Optional[FinishControl] = None
    problem: Problem = None
    solution_template: Optional[Solution] = None
    output_control: Optional[OutputControl] = None
    random_seed: Optional[int] = None
    additional_statistics_control: Optional[AdditionalStatisticsControl] = None


class AcoOptimizer(PopulationBasedMetaheuristic, metaclass=ABCMeta):
    """
    Instance of the class :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizer` encapsulates
    Ant Colony Optimization (MMAS variant) optimization algorithm.
    """

    def __init__(
        self,
        construction_support: "AcoConstructionSupport",
        evaporation_support: "AcoEvaporationSupport",
        n_ants: int,
        alpha: float,
        beta: float,
        rho: float,
        p_best: float,
        stagnation_limit: int,
        finish_control: FinishControl,
        problem: Problem,
        solution_template: Optional[Solution],
        output_control: Optional[OutputControl],
        random_seed: Optional[int],
        additional_statistics_control: Optional[AdditionalStatisticsControl]
    ) -> None:
        """
        Create new instance of class :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizer`.

        :param AcoConstructionSupport construction_support: builds ant solutions and applies local search
        :param AcoEvaporationSupport evaporation_support: pheromone evaporation, deposit and clamping strategy
        :param int n_ants: number of ants constructed per iteration
        :param float alpha: relative influence of pheromone trail during construction
        :param float beta: relative influence of heuristic desirability during construction
        :param float rho: base pheromone evaporation rate
        :param float p_best: MMAS parameter used to derive the pheromone lower bound
        :param int stagnation_limit: number of non-improving iterations before a pheromone reset
        :param FinishControl finish_control: structure that controls finish criteria for metaheuristic execution
        :param Problem problem: problem to be solved
        :param Optional[Solution] solution_template: initial solution of the problem
        :param Optional[OutputControl] output_control: structure that controls output
        :param Optional[int] random_seed: random seed for metaheuristic execution
        :param Optional[AdditionalStatisticsControl] additional_statistics_control: structure that controls additional
        statistics obtained during population-based metaheuristic execution
        """
        if not isinstance(n_ants, int):
            raise TypeError("Parameter 'n_ants' must be 'int'.")
        if n_ants <= 0:
            raise ValueError("Parameter 'n_ants' must be positive.")

        super().__init__(
            finish_control=finish_control,
            problem=problem,
            solution_template=solution_template,
            name="aco",
            output_control=output_control,
            random_seed=random_seed,
            additional_statistics_control=additional_statistics_control,
        )
        self.__construction_support = construction_support
        self.__evaporation_support = evaporation_support
        self.__n_ants = n_ants
        self.__alpha = alpha
        self.__beta = beta
        self.__rho = rho
        self.__p_best = p_best
        self.__stagnation_limit = stagnation_limit
        self.__stagnation_count = 0
        self.__tau: list[list[float]] = []
        self.__eta: list[list[float]] = []
        self.__tau_max: float = 1.0
        self.__tau_min: float = 0.0

    @abstractmethod
    def copy(self):
        """
        Copy the current object

        :return:  new instance with the same properties
        :rtype: :class:`AcoOptimizer`
        """
        raise NotImplementedError

    @property
    def construction_support(self) -> "AcoConstructionSupport":
        """
        Property getter for construction support.
        """
        return self.__construction_support

    @property
    def evaporation_support(self) -> "AcoEvaporationSupport":
        """
        Property getter for evaporation support.
        """
        return self.__evaporation_support

    @property
    def n_ants(self) -> int:
        """
        Property getter for number of ants.
        """
        return self.__n_ants

    @property
    def alpha(self) -> float:
        """
        Property getter for pheromone influence parameter.
        """
        return self.__alpha

    @property
    def beta(self) -> float:
        """
        Property getter for heuristic influence parameter.
        """
        return self.__beta

    @property
    def rho(self) -> float:
        """
        Property getter for base evaporation rate.
        """
        return self.__rho

    @property
    def p_best(self) -> float:
        """
        Property getter for the MMAS lower-bound parameter.
        """
        return self.__p_best

    @property
    def stagnation_limit(self) -> int:
        """
        Property getter for stagnation limit.
        """
        return self.__stagnation_limit

    @property
    def tau(self) -> list[list[float]]:
        """
        Property getter for the pheromone matrix.
        """
        return self.__tau

    @tau.setter
    def tau(self, value: list[list[float]]) -> None:
        """
        Property setter for the pheromone matrix.
        """
        self.__tau = value

    @property
    def eta(self) -> list[list[float]]:
        """
        Property getter for the heuristic desirability matrix.
        """
        return self.__eta

    @property
    def tau_max(self) -> float:
        """
        Property getter for the pheromone upper bound.
        """
        return self.__tau_max

    @property
    def tau_min(self) -> float:
        """
        Property getter for the pheromone lower bound.
        """
        return self.__tau_min

    @property
    def stagnation_count(self) -> int:
        """
        Property getter for stagnation count.
        """
        return self.__stagnation_count

    @stagnation_count.setter
    def stagnation_count(self, value: int) -> None:
        """
        Property setter for stagnation count.
        """
        self.__stagnation_count = value

    def update_tau_bounds(self, best_tour_cost: float) -> None:
        """
        Update the MMAS pheromone bounds from the given best tour cost.
        """
        if best_tour_cost <= 0:
            raise ValueError("Parameter 'best_tour_cost' must be positive.")
        if self.__rho <= 0 or self.__rho >= 1:
            raise ValueError("Parameter 'rho' must be in the interval (0, 1).")
        if not (0.0 < self.__p_best < 1.0):
            raise ValueError("Parameter 'p_best' must be in the interval (0, 1).")
        n = self.problem.n
        if n < 3:
            raise ValueError("MMAS pheromone bounds require problem size n >= 3.")
        self.__tau_max = 1.0 / (self.__rho * best_tour_cost)
        p = self.__p_best ** (1.0 / n)
        avg = n / 2
        denom = (avg - 1) * p
        if denom <= 0:
            raise ValueError("Invalid MMAS tau_min denominator; check 'p_best' and problem size 'n'.")
        self.__tau_min = self.__tau_max * (1 - p) / denom
        if self.__tau_min >= self.__tau_max:
            self.__tau_min = self.__tau_max * 1e-4

    def init(self) -> None:
        """
        Initialization of the ACO algorithm.
        """
        n = self.problem.n

        seed_solution = self.solution_template.copy()
        seed_solution.init_random(self.problem)
        seed_solution.evaluate(self.problem)
        seed_cost = seed_solution.tour_cost(self.problem, seed_solution.representation.tour)
        self.update_tau_bounds(seed_cost)

        self.__tau = [[self.__tau_max] * n for _ in range(n)]
        self.__eta = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.__eta[i][j] = 1.0 / self.problem.distances[i][j]

        self.best_solution = seed_solution
        self.evaluation = 1

    @abstractmethod
    def main_loop_iteration(self) -> None:
        """
        One iteration within main loop of the ACO algorithm
        """
        raise NotImplementedError

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the `AcoOptimizer` instance.
        """
        s = delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += group_start
        s += super().string_rep(delimiter, indentation, indentation_symbol, "", "")
        s += delimiter
        s += 'n_ants=' + str(self.n_ants) + delimiter
        s += 'alpha=' + str(self.alpha) + delimiter
        s += 'beta=' + str(self.beta) + delimiter
        s += 'rho=' + str(self.rho) + delimiter
        s += group_end
        return s

    def __str__(self) -> str:
        """
        String representation of the object.
        """
        return self.string_rep('|')

    def __repr__(self) -> str:
        """
        Representation of the object.
        """
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        """
        Formatted representation of the object.
        """
        return self.string_rep('\n', 0, '   ', '{', '}')
