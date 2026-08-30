"""
..  _py_aco_optimizer_standard:
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from typing import Optional

from uo.problem.problem import Problem
from uo.solution.solution import Solution

from uo.algorithm.output_control import OutputControl
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.additional_statistics_control import AdditionalStatisticsControl

from uo.algorithm.metaheuristic.aco.aco_optimizer import AcoOptimizer, AcoOptimizerConstructionParameters


class AcoOptimizerStandard(AcoOptimizer):
    """
    Instance of the class :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizerStandard` encapsulates
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
        output_control: Optional[OutputControl] = None,
        random_seed: Optional[int] = None,
        additional_statistics_control: Optional[AdditionalStatisticsControl] = None
    ) -> None:
        """
        Create new instance of class :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizerStandard`.
        """
        super().__init__(
            construction_support=construction_support,
            evaporation_support=evaporation_support,
            n_ants=n_ants,
            alpha=alpha,
            beta=beta,
            rho=rho,
            p_best=p_best,
            stagnation_limit=stagnation_limit,
            finish_control=finish_control,
            problem=problem,
            solution_template=solution_template,
            output_control=output_control,
            random_seed=random_seed,
            additional_statistics_control=additional_statistics_control,
        )

    def copy(self) -> "AcoOptimizerStandard":
        """
        Internal copy of the current instance.

        :return: new instance of class :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizerStandard`
        """
        return AcoOptimizerStandard(
            construction_support=self.construction_support,
            evaporation_support=self.evaporation_support,
            n_ants=self.n_ants,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            p_best=self.p_best,
            stagnation_limit=self.stagnation_limit,
            finish_control=self.finish_control.copy(),
            problem=self.problem.copy(),
            solution_template=self.solution_template.copy() if self.solution_template else None,
            output_control=self.output_control,
            random_seed=self.random_seed,
            additional_statistics_control=self.additional_statistics_control,
        )

    @classmethod
    def from_construction_tuple(cls, construction_tuple: AcoOptimizerConstructionParameters) -> "AcoOptimizerStandard":
        """
        Additional constructor, that creates new instance of class
        :class:`~uo.algorithm.metaheuristic.aco.AcoOptimizerStandard`.

        :param AcoOptimizerConstructionParameters construction_tuple: tuple with all constructor parameters
        """
        return cls(
            construction_support=construction_tuple.construction_support,
            evaporation_support=construction_tuple.evaporation_support,
            n_ants=construction_tuple.n_ants,
            alpha=construction_tuple.alpha,
            beta=construction_tuple.beta,
            rho=construction_tuple.rho,
            p_best=construction_tuple.p_best,
            stagnation_limit=construction_tuple.stagnation_limit,
            finish_control=construction_tuple.finish_control,
            problem=construction_tuple.problem,
            solution_template=construction_tuple.solution_template,
            output_control=construction_tuple.output_control,
            random_seed=construction_tuple.random_seed,
            additional_statistics_control=construction_tuple.additional_statistics_control,
        )

    def main_loop_iteration(self) -> None:
        """
        One iteration within main loop of the ACO algorithm.
        """
        self.iteration += 1

        iteration_best_solution = None
        for _ in range(self.n_ants):
            ant_solution = self.solution_template.copy()
            self.construction_support.construct(self.problem, ant_solution, self)
            ant_solution.evaluate(self.problem)
            self.evaluation += 1
            if (
                iteration_best_solution is None
                or ant_solution.fitness_value > iteration_best_solution.fitness_value
            ):
                iteration_best_solution = ant_solution

        self.construction_support.local_search(self.problem, iteration_best_solution, self)
        iteration_best_solution.evaluate(self.problem)
        self.evaluation += 1

        improved = iteration_best_solution.fitness_value > self.best_solution.fitness_value
        if improved:
            cost = iteration_best_solution.tour_cost(
                self.problem, iteration_best_solution.representation.tour
            )
            self.update_tau_bounds(cost)
            self.stagnation_count = 0
            self.best_solution = iteration_best_solution
        else:
            self.stagnation_count += 1

        if self.stagnation_count >= self.stagnation_limit:
            n = self.problem.n
            self.tau = [[self.tau_max] * n for _ in range(n)]
            self.stagnation_count = 0

        self.evaporation_support.apply(self, iteration_best_solution, self.best_solution)
