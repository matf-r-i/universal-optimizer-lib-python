"""Tolerance based check that BFO actually optimizes a real objective function."""

from datetime import datetime
from random import Random
import unittest

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_movement_support import (
    BfoMovementSupportReal,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_optimizer import BfoOptimizer
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_step_size_support import (
    BfoStepSizeSupportAdaptive,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_swarming_support import (
    BfoSwarmingSupportIdle,
)
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.quality_of_solution import QualityOfSolution
from uo.solution.solution import Solution

LOWER_BOUNDS = (-5.0, -5.0)
UPPER_BOUNDS = (5.0, 5.0)


class SphereSolution(Solution):
    """Two dimensional sphere function with its optimum in the coordinate origin

    Fitness is the negated objective, because the optimizer always treats a
    larger fitness value as the better one
    """

    def __init__(self, representation=(0.0, 0.0), random_generator=None):
        super().__init__(
            random_seed=1,
            fitness_value=None,
            fitness_values=None,
            objective_value=None,
            objective_values=None,
            is_feasible=True,
        )
        self.representation = tuple(float(value) for value in representation)
        self.__random_generator = random_generator or Random(1234)

    def copy(self):
        copied = SphereSolution(self.representation, self.__random_generator)
        copied.fitness_value = self.fitness_value
        copied.objective_value = self.objective_value
        copied.is_feasible = self.is_feasible
        return copied

    def copy_from(self, original):
        super().copy_from(original)

    def argument(self, representation):
        return str(representation)

    def init_random(self, problem):
        self.representation = tuple(
            self.__random_generator.uniform(lower, upper)
            for lower, upper in zip(LOWER_BOUNDS, UPPER_BOUNDS)
        )

    def native_representation(self, representation_str):
        return tuple(float(value) for value in representation_str.strip("()").split(","))

    def init_from(self, representation, problem):
        self.representation = tuple(float(value) for value in representation)

    def calculate_quality_directly(self, representation, problem):
        objective_value = sum(value * value for value in representation)
        return QualityOfSolution(
            objective_value=objective_value,
            objective_values=None,
            fitness_value=-objective_value,
            fitness_values=None,
            is_feasible=True,
        )

    def representation_distance_directly(self, representation_1, representation_2):
        return sum(
            abs(first - second)
            for first, second in zip(representation_1, representation_2)
        )

    def __str__(self):
        return self.string_rep("|")

    def __repr__(self):
        return self.string_rep("\n")

    def __format__(self, spec):
        return str(self)


class TestBfoOptimizerConvergence(unittest.TestCase):

    def make_optimizer(self, random_seed=17):
        optimizer = BfoOptimizer(
            movement_support=BfoMovementSupportReal(LOWER_BOUNDS, UPPER_BOUNDS),
            swarming_support=BfoSwarmingSupportIdle(),
            step_size_support=BfoStepSizeSupportAdaptive(
                initial_step_size=0.5,
                minimum_step_size=0.001,
                maximum_step_size=1.0,
                increase_factor=1.1,
                decrease_factor=0.5,
            ),
            population_size=10,
            chemotactic_steps=10,
            swim_length=4,
            reproduction_steps=4,
            elimination_dispersal_events=2,
            elimination_dispersal_probability=0.1,
            finish_control=FinishControl(criteria=""),
            problem=ProblemVoidMinSO("sphere"),
            solution_template=SphereSolution(random_generator=Random(random_seed)),
            random_seed=random_seed,
        )
        optimizer.execution_started = datetime.now()
        return optimizer

    def test_optimizer_approaches_the_sphere_optimum(self):
        optimizer = self.make_optimizer()

        optimizer.init()
        objective_after_init = optimizer.best_solution.objective_value

        result = optimizer.optimize()

        self.assertLess(result.objective_value, objective_after_init)
        self.assertLess(result.objective_value, 0.01)
        for value, lower, upper in zip(result.representation, LOWER_BOUNDS, UPPER_BOUNDS):
            self.assertGreaterEqual(value, lower)
            self.assertLessEqual(value, upper)

    def test_optimizer_converges_for_every_tried_seed(self):
        for random_seed in (1, 5, 17, 42, 101):
            with self.subTest(random_seed=random_seed):
                optimizer = self.make_optimizer(random_seed=random_seed)

                result = optimizer.optimize()

                self.assertLess(result.objective_value, 0.1)


if __name__ == "__main__":
    unittest.main()
