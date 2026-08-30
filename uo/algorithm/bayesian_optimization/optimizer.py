"""Universal Optimizer adapter for Gaussian-process Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from uo.algorithm.algorithm import Algorithm
from uo.algorithm.bayesian_optimization.acquisition_optimization import suggest_next_point
from uo.algorithm.bayesian_optimization.gaussian_process import GaussianProcessRegressor
from uo.algorithm.bayesian_optimization.kernels import RBFKernel
from uo.algorithm.bayesian_optimization.space import (
    from_unit_cube,
    sample_uniform,
    to_unit_cube,
    validate_bounds,
)
from uo.algorithm.bayesian_optimization.types import FloatArray
from uo.algorithm.output_control import OutputControl
from uo.problem.problem import Problem
from uo.solution.solution import Solution


@dataclass(frozen=True)
class GaussianProcessConfig:
    """Configuration of the Gaussian-process surrogate."""

    length_scale: float = 0.5
    amplitude: float = 1.0
    mean_value: float = 0.0
    jitter: float = 1e-8

    def __post_init__(self) -> None:
        RBFKernel(self.length_scale, self.amplitude)
        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")


@dataclass(frozen=True)
class AcquisitionConfig:
    """Configuration of Expected Improvement optimization."""

    xi: float = 0.01
    number_of_restarts: int = 8

    def __post_init__(self) -> None:
        if self.xi < 0.0:
            raise ValueError("xi must be non-negative.")
        if self.number_of_restarts <= 0:
            raise ValueError("number_of_restarts must be positive.")


@dataclass
class BayesianOptimizerConstructionParameters:
    """Construction parameters for :class:`BayesianOptimizer`."""

    problem: Optional[Problem] = None
    solution_template: Optional[Solution] = None
    bounds: Sequence[tuple[float, float]] | FloatArray = field(default_factory=list)
    evaluation_budget: int = 0
    number_of_initial_points: Optional[int] = None
    random_seed: Optional[int] = None
    gaussian_process_config: GaussianProcessConfig = field(default_factory=GaussianProcessConfig)
    acquisition_config: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    output_control: Optional[OutputControl] = None


class BayesianOptimizer(Algorithm):
    """Bayesian optimizer for bounded, continuous, single-objective problems."""

    def __init__(
        self,
        problem: Problem,
        solution_template: Solution,
        bounds: Sequence[tuple[float, float]] | FloatArray,
        evaluation_budget: int,
        number_of_initial_points: Optional[int] = None,
        random_seed: Optional[int] = None,
        gaussian_process_config: Optional[GaussianProcessConfig] = None,
        acquisition_config: Optional[AcquisitionConfig] = None,
        output_control: Optional[OutputControl] = None,
    ) -> None:
        if solution_template is None:
            raise TypeError("solution_template must be a Solution.")
        if problem.is_multi_objective:
            raise ValueError("BayesianOptimizer supports only single-objective problems.")
        if not isinstance(evaluation_budget, int):
            raise TypeError("evaluation_budget must be int.")
        if evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive.")
        if number_of_initial_points is not None and number_of_initial_points <= 0:
            raise ValueError("number_of_initial_points must be positive.")
        if random_seed is not None and not isinstance(random_seed, int):
            raise TypeError("random_seed must be int or None.")

        super().__init__(
            problem=problem,
            solution_template=solution_template,
            name="BayesianOptimization",
            output_control=output_control,
        )
        self._bounds = validate_bounds(bounds)
        self._evaluation_budget = evaluation_budget
        self._number_of_initial_points = number_of_initial_points
        self._random_seed = random_seed
        self._gaussian_process_config = gaussian_process_config or GaussianProcessConfig()
        self._acquisition_config = acquisition_config or AcquisitionConfig()
        self._rng = np.random.default_rng(random_seed)
        self._x_history = np.empty((0, self._bounds.shape[0]), dtype=np.float64)
        self._target_history = np.empty(0, dtype=np.float64)
        self._acquisition_history = np.empty(0, dtype=np.float64)
        self._unit_bounds = np.column_stack((
            np.zeros(self._bounds.shape[0], dtype=np.float64),
            np.ones(self._bounds.shape[0], dtype=np.float64),
        ))

    @classmethod
    def from_construction_tuple(
        cls,
        construction_parameters: BayesianOptimizerConstructionParameters,
    ) -> BayesianOptimizer:
        """Create an optimizer from a construction-parameter object."""
        return cls(
            problem=construction_parameters.problem,
            solution_template=construction_parameters.solution_template,
            bounds=construction_parameters.bounds,
            evaluation_budget=construction_parameters.evaluation_budget,
            number_of_initial_points=construction_parameters.number_of_initial_points,
            random_seed=construction_parameters.random_seed,
            gaussian_process_config=construction_parameters.gaussian_process_config,
            acquisition_config=construction_parameters.acquisition_config,
            output_control=construction_parameters.output_control,
        )

    @property
    def bounds(self) -> FloatArray:
        """Return a copy of the optimization bounds."""
        return self._bounds.copy()

    @property
    def evaluation_budget(self) -> int:
        """Return the maximum number of objective evaluations."""
        return self._evaluation_budget

    @property
    def x_history(self) -> FloatArray:
        """Return a copy of all evaluated vectors."""
        return self._x_history.copy()

    @property
    def target_history(self) -> FloatArray:
        """Return minimized GP targets, equal to negative fitness values."""
        return self._target_history.copy()

    @property
    def acquisition_history(self) -> FloatArray:
        """Return acquisition values for model-selected candidates."""
        return self._acquisition_history.copy()

    def copy(self) -> BayesianOptimizer:
        """Copy optimizer configuration without execution state."""
        return BayesianOptimizer(
            problem=self.problem.copy(),
            solution_template=self.solution_template.copy(),
            bounds=self._bounds.copy(),
            evaluation_budget=self._evaluation_budget,
            number_of_initial_points=self._number_of_initial_points,
            random_seed=self._random_seed,
            gaussian_process_config=self._gaussian_process_config,
            acquisition_config=self._acquisition_config,
            output_control=self.output_control.copy() if self.output_control is not None else None,
        )

    def init(self) -> None:
        """Reset execution state and evaluate the initial random design."""
        self.evaluation = 0
        self.iteration = 0
        self.evaluation_best_found = 0
        self.iteration_best_found = 0
        self._rng = np.random.default_rng(self._random_seed)
        self._x_history = np.empty((0, self._bounds.shape[0]), dtype=np.float64)
        self._target_history = np.empty(0, dtype=np.float64)
        self._acquisition_history = np.empty(0, dtype=np.float64)

        default_initial = max(5, 2 * self._bounds.shape[0])
        initial_count = default_initial if self._number_of_initial_points is None else self._number_of_initial_points
        initial_count = min(initial_count, self._evaluation_budget)
        for point in sample_uniform(self._rng, self._bounds, initial_count):
            self._evaluate_point(point)

    def optimize(self) -> Solution:
        """Run Bayesian optimization and return the best evaluated solution."""
        self.execution_started = datetime.now()
        self.init()
        self.write_output_headers_if_needed()
        self.write_output_values_if_needed("before_algorithm", "b_a")
        gaussian_process = self._build_gaussian_process()

        while self.evaluation < self._evaluation_budget:
            self.write_output_values_if_needed("before_iteration", "b_i")
            self.iteration += 1
            gaussian_process.fit(to_unit_cube(self._x_history, self._bounds), self._target_history)
            next_unit, acquisition_value = suggest_next_point(
                gp=gaussian_process,
                bounds=self._unit_bounds,
                best_y=float(np.min(self._target_history)),
                xi=self._acquisition_config.xi,
                rng=self._rng,
                number_of_restarts=self._acquisition_config.number_of_restarts,
            )
            next_point = np.asarray(from_unit_cube(next_unit, self._bounds), dtype=np.float64)
            next_point = self._ensure_novel_candidate(next_point)
            self._evaluate_point(next_point)
            self._acquisition_history = np.append(self._acquisition_history, acquisition_value)
            self.write_output_values_if_needed("after_iteration", "a_i")

        self.execution_ended = datetime.now()
        self.write_output_values_if_needed("after_algorithm", "a_a")
        return self.best_solution

    def _evaluate_point(self, point: FloatArray) -> None:
        candidate = self.solution_template.copy()
        candidate.init_from(np.asarray(point, dtype=np.float64).copy(), self.problem)
        self.write_output_values_if_needed("before_evaluation", "b_e")
        candidate.evaluate(self.problem)
        self.evaluation += 1
        if not candidate.is_feasible:
            raise ValueError("BayesianOptimizer does not support infeasible evaluations.")
        if candidate.fitness_value is None or not np.isfinite(candidate.fitness_value):
            raise ValueError("candidate fitness must be a finite number.")

        self._x_history = np.vstack((self._x_history, point.reshape(1, -1)))
        self._target_history = np.append(self._target_history, -float(candidate.fitness_value))
        if self.best_solution is None or candidate.is_better(self.best_solution, self.problem):
            self.best_solution = candidate
        self.write_output_values_if_needed("after_evaluation", "a_e")

    def _build_gaussian_process(self) -> GaussianProcessRegressor:
        config = self._gaussian_process_config
        return GaussianProcessRegressor(
            kernel=RBFKernel(config.length_scale, config.amplitude),
            mean_value=config.mean_value,
            jitter=config.jitter,
        )

    def _ensure_novel_candidate(
        self,
        candidate: FloatArray,
        tolerance: float = 1e-10,
        maximum_attempts: int = 64,
    ) -> FloatArray:
        if self._is_novel(candidate, tolerance):
            return candidate
        for _ in range(maximum_attempts):
            sampled = sample_uniform(self._rng, self._bounds, 1)[0]
            if self._is_novel(sampled, tolerance):
                return sampled
        return candidate

    def _is_novel(self, candidate: FloatArray, tolerance: float) -> bool:
        if self._x_history.shape[0] == 0:
            return True
        differences = np.max(np.abs(self._x_history - candidate[None, :]), axis=1)
        return bool(np.all(differences > tolerance))

    def string_rep(
        self,
        delimiter: str,
        indentation: int = 0,
        indentation_symbol: str = "",
        group_start: str = "{",
        group_end: str = "}",
    ) -> str:
        """Return a string representation of the optimizer."""
        result = super().string_rep(delimiter, indentation, indentation_symbol, group_start, "")
        result += delimiter + "bounds=" + str(self._bounds.tolist())
        result += delimiter + "evaluation_budget=" + str(self._evaluation_budget)
        result += delimiter + group_end
        return result

    def __str__(self) -> str:
        return self.string_rep("|")

    def __repr__(self) -> str:
        return self.string_rep("\n")

    def __format__(self, spec: str) -> str:
        return self.string_rep("|")
