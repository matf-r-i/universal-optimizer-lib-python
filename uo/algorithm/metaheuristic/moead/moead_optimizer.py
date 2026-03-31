"""
..  _py_moead_optimizer:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_optimizer`
module contains class
:class:`~uo.algorithm.metaheuristic.moead.moead_optimizer.MoeadOptimizer`,
that represents implementation of the
:ref:`MOEA/D<Algorithm_MOEAD>` algorithm.

The algorithm decomposes a multi-objective optimization problem into a
set of scalar subproblems defined by weight vectors and solves them
cooperatively by means of neighborhood-based offspring generation and
replacement.
"""

from __future__ import annotations

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))

from random import Random
from typing import Optional

import numpy as np

from uo.problem.problem import Problem
from uo.solution.solution import Solution

from uo.algorithm.output_control import OutputControl
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.additional_statistics_control import AdditionalStatisticsControl
from uo.algorithm.metaheuristic.population_based_metaheuristic import PopulationBasedMetaheuristic

from uo.algorithm.metaheuristic.moead.moead_decomposition import tchebyscheff_one
from uo.algorithm.metaheuristic.moead.moead_metrics import filter_nondominated_points
from uo.algorithm.metaheuristic.moead.moead_variation_support import MoeadVariationSupport
from uo.algorithm.metaheuristic.moead.moead_weights import WeightSetup, build_weight_setup


class MoeadOptimizer(PopulationBasedMetaheuristic):
    """
    Implementation of the Multi-Objective Evolutionary Algorithm based on
    Decomposition (MOEA/D).
    """

    def __init__(
            self,
            moead_variation_support: MoeadVariationSupport,
            population_size: int,
            neighborhood_size: int,
            max_number_of_replaced_neighbors: int,
            finish_control: FinishControl,
            problem: Problem,
            solution_template: Optional[Solution],
            output_control: Optional[OutputControl],
            random_seed: Optional[int],
            additional_statistics_control: Optional[AdditionalStatisticsControl],
            lattice_parameter_H: Optional[int] = None
    ) -> None:
        """
        Create new instance of MOEA/D optimizer.

        :param MoeadVariationSupport moead_variation_support: support object used for offspring generation
        :param int population_size: size of the population
        :param int neighborhood_size: neighborhood size
        :param int max_number_of_replaced_neighbors: maximal number of neighbors replaced by one offspring
        :param FinishControl finish_control: finish control
        :param Problem problem: target problem
        :param Optional[Solution] solution_template: solution template
        :param Optional[OutputControl] output_control: output control
        :param Optional[int] random_seed: random seed
        :param Optional[AdditionalStatisticsControl] additional_statistics_control: additional statistics control
        :param Optional[int] lattice_parameter_H: simplex-lattice parameter for problems with three or more objectives
        """
        if not isinstance(moead_variation_support, MoeadVariationSupport):
            raise TypeError("Parameter 'moead_variation_support' must be 'MoeadVariationSupport'.")
        if not isinstance(population_size, int):
            raise TypeError("Parameter 'population_size' must be 'int'.")
        if population_size <= 0:
            raise ValueError("Parameter 'population_size' must be positive.")
        if not isinstance(neighborhood_size, int):
            raise TypeError("Parameter 'neighborhood_size' must be 'int'.")
        if neighborhood_size <= 0:
            raise ValueError("Parameter 'neighborhood_size' must be positive.")
        if not isinstance(max_number_of_replaced_neighbors, int):
            raise TypeError("Parameter 'max_number_of_replaced_neighbors' must be 'int'.")
        if max_number_of_replaced_neighbors <= 0:
            raise ValueError("Parameter 'max_number_of_replaced_neighbors' must be positive.")
        if not isinstance(problem, Problem):
            raise TypeError("Parameter 'problem' must be 'Problem'.")
        if not problem.is_multi_objective:
            raise ValueError("MOEA/D can be used only for multi-objective problems.")
        if solution_template is None:
            raise ValueError("Parameter 'solution_template' must not be None.")

        super().__init__(
            finish_control=finish_control,
            problem=problem,
            solution_template=solution_template,
            name='moead',
            output_control=output_control,
            random_seed=random_seed,
            additional_statistics_control=additional_statistics_control
        )

        self.__moead_variation_support: MoeadVariationSupport = moead_variation_support
        self.__population_size: int = population_size
        self.__neighborhood_size: int = neighborhood_size
        self.__max_number_of_replaced_neighbors: int = max_number_of_replaced_neighbors
        self.__lattice_parameter_H: Optional[int] = lattice_parameter_H

        self.__current_population: list[Solution] = []
        for _ in range(self.population_size):
            self.__current_population.append(self.solution_template.copy())

        self.__weight_setup: Optional[WeightSetup] = None
        self.__ideal_point: Optional[np.ndarray] = None
        self.__nondominated_archive: list[Solution] = []

        self.__random_generator = Random(self.random_seed)

    def copy(self) -> "MoeadOptimizer":
        """
        Copy the current object.

        :return: copied MOEA/D optimizer
        :rtype: MoeadOptimizer
        """
        new_obj = MoeadOptimizer(
            moead_variation_support=self.moead_variation_support.copy(),
            population_size=self.population_size,
            neighborhood_size=self.neighborhood_size,
            max_number_of_replaced_neighbors=self.max_number_of_replaced_neighbors,
            finish_control=self.finish_control.copy(),
            problem=self.problem.copy(),
            solution_template=self.solution_template.copy(),
            output_control=self.output_control,
            random_seed=self.random_seed,
            additional_statistics_control=self.additional_statistics_control,
            lattice_parameter_H=self.lattice_parameter_H
        )

        copied_population: list[Solution] = []
        for sol in self.current_population:
            copied_population.append(sol.copy())
        new_obj.current_population = copied_population

        if self.__ideal_point is not None:
            new_obj.__ideal_point = self.__ideal_point.copy()

        copied_archive: list[Solution] = []
        for sol in self.__nondominated_archive:
            copied_archive.append(sol.copy())
        new_obj.__nondominated_archive = copied_archive

        if self.best_solution is not None:
            new_obj.best_solution = self.best_solution.copy()

        new_obj.evaluation = self.evaluation
        new_obj.iteration = self.iteration
        new_obj.evaluation_best_found = self.evaluation_best_found
        new_obj.iteration_best_found = self.iteration_best_found

        return new_obj

    @property
    def population_size(self) -> int:
        """
        Property getter for population size.
        """
        return self.__population_size

    @property
    def neighborhood_size(self) -> int:
        """
        Property getter for neighborhood size.
        """
        return self.__neighborhood_size

    @property
    def max_number_of_replaced_neighbors(self) -> int:
        """
        Property getter for maximum number of replaced neighbors.
        """
        return self.__max_number_of_replaced_neighbors

    @property
    def lattice_parameter_H(self) -> Optional[int]:
        """
        Property getter for simplex-lattice parameter.
        """
        return self.__lattice_parameter_H

    @property
    def moead_variation_support(self) -> MoeadVariationSupport:
        """
        Property getter for variation support.
        """
        return self.__moead_variation_support

    @property
    def current_population(self) -> list[Solution]:
        """
        Property getter for current population.
        """
        return self.__current_population

    @current_population.setter
    def current_population(self, value: list[Solution]) -> None:
        """
        Property setter for current population.
        """
        if not isinstance(value, list):
            raise TypeError("Parameter 'current_population' must have type 'list'.")
        self.__population_size = len(value)
        self.__current_population = value

    @property
    def weight_setup(self) -> Optional[WeightSetup]:
        """
        Property getter for weight setup.
        """
        return self.__weight_setup

    @property
    def ideal_point(self) -> Optional[np.ndarray]:
        """
        Property getter for ideal point.
        """
        return self.__ideal_point

    @property
    def nondominated_archive(self) -> list[Solution]:
        """
        Property getter for nondominated archive.
        """
        return self.__nondominated_archive

    def _objective_vector_of_solution(self, solution: Solution) -> np.ndarray:
        """
        Obtain objective vector from a solution.

        :param Solution solution: target solution
        :return: objective vector
        :rtype: np.ndarray
        """
        if solution.objective_values is None:
            raise ValueError("Solution objective_values must not be None for MOEA/D.")
        return np.asarray(solution.objective_values, dtype=float)

    def _objective_matrix_of_population(self) -> np.ndarray:
        """
        Build objective matrix from current population.

        :return: objective matrix of shape ``(N, M)``
        :rtype: np.ndarray
        """
        return np.vstack([self._objective_vector_of_solution(sol) for sol in self.current_population])

    def _update_ideal_point_with_solution(self, solution: Solution) -> None:
        """
        Update ideal point using objective values of one solution.

        :param Solution solution: solution used to update ideal point
        """
        f = self._objective_vector_of_solution(solution)
        if self.__ideal_point is None:
            self.__ideal_point = f.copy()
        else:
            self.__ideal_point = np.minimum(self.__ideal_point, f)

    def _recompute_ideal_point_from_population(self) -> None:
        """
        Recompute ideal point from the whole population.
        """
        F = self._objective_matrix_of_population()
        self.__ideal_point = np.min(F, axis=0)

    def _dominates(self, s1: Solution, s2: Solution) -> bool:
        """
        Check whether solution ``s1`` dominates solution ``s2`` for minimization.

        :param Solution s1: first solution
        :param Solution s2: second solution
        :return: domination indicator
        :rtype: bool
        """
        f1 = self._objective_vector_of_solution(s1)
        f2 = self._objective_vector_of_solution(s2)
        return bool(np.all(f1 <= f2) and np.any(f1 < f2))

    def _update_nondominated_archive(self) -> None:
        """
        Rebuild nondominated archive from current population.
        """
        archive: list[Solution] = []

        for candidate in self.current_population:
            dominated = False
            to_remove: list[int] = []

            for idx, existing in enumerate(archive):
                if self._dominates(existing, candidate):
                    dominated = True
                    break
                if self._dominates(candidate, existing):
                    to_remove.append(idx)

            if dominated:
                continue

            for idx in reversed(to_remove):
                archive.pop(idx)

            archive.append(candidate.copy())

        self.__nondominated_archive = archive

    def _update_representative_best_solution(self) -> None:
        """
        Update representative best solution.

        Since generic ``Solution.is_better`` is not intended for
        multi-objective comparison, MOEA/D keeps one representative
        solution only for framework compatibility and logging.
        """
        if len(self.current_population) == 0:
            return

        if self.__ideal_point is None:
            self._recompute_ideal_point_from_population()

        if self.__weight_setup is None:
            raise ValueError("Weight setup must not be None.")

        reference_weight = np.mean(self.__weight_setup.W, axis=0)

        best_idx = 0
        best_val = tchebyscheff_one(
            self._objective_vector_of_solution(self.current_population[0]),
            reference_weight,
            self.__ideal_point
        )

        for i in range(1, self.population_size):
            val = tchebyscheff_one(
                self._objective_vector_of_solution(self.current_population[i]),
                reference_weight,
                self.__ideal_point
            )
            if val < best_val:
                best_val = val
                best_idx = i

        self.best_solution = self.current_population[best_idx].copy()
        self.best_solution.fitness_value = float(-best_val)
        self.best_solution.objective_value = float(best_val)

    def _replace_neighbors(
            self,
            offspring: Solution,
            neighbor_indices: np.ndarray
    ) -> int:
        """
        Try to replace neighboring subproblem solutions using one offspring.

        :param Solution offspring: offspring solution
        :param np.ndarray neighbor_indices: neighborhood indices
        :return: number of replacements
        :rtype: int
        """
        if self.__weight_setup is None:
            raise ValueError("Weight setup must not be None.")
        if self.__ideal_point is None:
            raise ValueError("Ideal point must not be None.")

        f_y = self._objective_vector_of_solution(offspring)

        replaced = 0
        for j in neighbor_indices:
            current = self.current_population[int(j)]
            f_old = self._objective_vector_of_solution(current)

            g_old = tchebyscheff_one(f_old, self.__weight_setup.W[int(j)], self.__ideal_point)
            g_new = tchebyscheff_one(f_y, self.__weight_setup.W[int(j)], self.__ideal_point)

            if g_new <= g_old:
                self.current_population[int(j)] = offspring.copy()
                replaced += 1

                if replaced >= self.max_number_of_replaced_neighbors:
                    break

        return replaced

    def init(self) -> None:
        """
        Initialization of the MOEA/D algorithm.
        """
        for i in range(self.population_size):
            self.current_population[i].init_random(self.problem)
            self.evaluation += 1
            self.current_population[i].evaluate(self.problem)

        n_obj = len(self._objective_vector_of_solution(self.current_population[0]))

        self.__weight_setup = build_weight_setup(
            n_obj=n_obj,
            population_size=self.population_size,
            H=self.lattice_parameter_H,
            T=self.neighborhood_size
        )

        self._recompute_ideal_point_from_population()
        self._update_nondominated_archive()
        self._update_representative_best_solution()

    def main_loop_iteration(self) -> None:
        """
        One iteration within main loop of the MOEA/D algorithm.
        """
        if self.__weight_setup is None:
            raise ValueError("Weight setup must not be None.")
        if self.__ideal_point is None:
            raise ValueError("Ideal point must not be None.")

        for i in range(self.population_size):
            neighborhood = self.__weight_setup.B[i]

            if len(neighborhood) >= 2:
                parent_indices = self.__random_generator.sample(list(neighborhood), 2)
            else:
                parent_indices = [int(neighborhood[0]), int(neighborhood[0])]

            parent1 = self.current_population[parent_indices[0]]
            parent2 = self.current_population[parent_indices[1]]

            offspring = self.solution_template.copy()

            self.moead_variation_support.generate_offspring(
                problem=self.problem,
                parent1=parent1,
                parent2=parent2,
                child=offspring,
                optimizer=self
            )

            self._update_ideal_point_with_solution(offspring)
            self._replace_neighbors(offspring, neighborhood)

        self.iteration += 1
        self._update_nondominated_archive()
        self._update_representative_best_solution()

    def string_rep(
            self,
            delimiter: str,
            indentation: int = 0,
            indentation_symbol: str = '',
            group_start: str = '{',
            group_end: str = '}'
    ) -> str:
        """
        String representation of the MOEA/D optimizer.
        """
        s = delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += group_start
        s = super().string_rep(delimiter, indentation, indentation_symbol, '', '')
        s += delimiter
        s += 'current_population=' + group_start
        if self.__current_population is not None:
            for individual in self.__current_population:
                s += individual.string_rep(delimiter, indentation + 1,
                                           indentation_symbol, group_start, group_end) + delimiter
            s += group_end
        else:
            s += 'None'
        s += delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += 'population_size=' + str(self.population_size) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += 'neighborhood_size=' + str(self.neighborhood_size) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += 'max_number_of_replaced_neighbors=' + str(self.max_number_of_replaced_neighbors) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += 'moead_variation_support=' + self.moead_variation_support.string_rep(
            delimiter, indentation + 1, indentation_symbol, group_start, group_end
        ) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
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
        return self.string_rep('|')