from datetime import datetime
import unittest

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_movement_support import (
    BfoMovementSupport,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_optimizer import (
    BfoOptimizer,
    BfoOptimizerConstructionParameters,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_step_size_support import (
    BfoStepSizeSupportFixed,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_swarming_support import (
    BfoSwarmingSupportIdle,
)
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.quality_of_solution import QualityOfSolution
from uo.solution.solution import Solution


class LinearSolution(Solution):
    """Small deterministic solution used to exercise the BFO lifecycle."""

    def __init__(self, representation=0.0):
        super().__init__(
            random_seed=1,
            fitness_value=None,
            fitness_values=None,
            objective_value=None,
            objective_values=None,
            is_feasible=True,
        )
        self.representation = float(representation)

    def copy(self):
        copied = LinearSolution(self.representation)
        copied.fitness_value = self.fitness_value
        copied.objective_value = self.objective_value
        copied.is_feasible = self.is_feasible
        return copied

    def copy_from(self, original):
        super().copy_from(original)

    def argument(self, representation):
        return str(representation)

    def init_random(self, problem):
        self.representation = 0.0

    def native_representation(self, representation_str):
        return float(representation_str)

    def init_from(self, representation, problem):
        self.representation = float(representation)

    def calculate_quality_directly(self, representation, problem):
        return QualityOfSolution(
            objective_value=representation,
            objective_values=None,
            fitness_value=representation,
            fitness_values=None,
            is_feasible=True,
        )

    def representation_distance_directly(self, representation_1, representation_2):
        return abs(representation_1 - representation_2)

    def __str__(self):
        return self.string_rep("|")

    def __repr__(self):
        return self.string_rep("\n")

    def __format__(self, spec):
        return str(self)


class DeterministicMovement(BfoMovementSupport):
    def __init__(self):
        self.move_calls = 0

    def copy(self):
        return DeterministicMovement()

    def tumble_direction(self, solution, random_generator):
        return 1.0

    def move(self, solution, direction, step_size, problem):
        self.move_calls += 1
        candidate = solution.copy()
        candidate.init_from(solution.representation + direction * step_size, problem)
        return candidate


class TestBfoOptimizer(unittest.TestCase):

    def make_optimizer(self, **overrides):
        arguments = {
            "movement_support": DeterministicMovement(),
            "swarming_support": BfoSwarmingSupportIdle(),
            "step_size_support": BfoStepSizeSupportFixed(0.5),
            "population_size": 4,
            "chemotactic_steps": 2,
            "swim_length": 1,
            "reproduction_steps": 2,
            "elimination_dispersal_events": 1,
            "elimination_dispersal_probability": 1.0,
            "finish_control": FinishControl(criteria=""),
            "problem": ProblemVoidMinSO("linear"),
            "solution_template": LinearSolution(),
            "random_seed": 7,
        }
        arguments.update(overrides)
        optimizer = BfoOptimizer(**arguments)
        optimizer.execution_started = datetime.now()
        return optimizer

    def test_init_creates_population_and_fixed_step_sizes(self):
        optimizer = self.make_optimizer()

        optimizer.init()

        self.assertEqual(len(optimizer.current_population), 4)
        self.assertEqual(optimizer.evaluation, 4)
        self.assertEqual(optimizer.step_sizes, [0.5] * 4)
        self.assertEqual(optimizer.health, [0.0] * 4)
        self.assertIsNotNone(optimizer.best_solution)

    def test_chemotaxis_uses_fixed_size_and_idle_swarming(self):
        optimizer = self.make_optimizer(chemotactic_steps=1, swim_length=1)
        optimizer.init()
        movement = optimizer.movement_support

        optimizer.main_loop_iteration()

        self.assertEqual(movement.move_calls, 8)
        self.assertEqual(optimizer.evaluation, 12)
        self.assertEqual(optimizer.step_sizes, [0.5] * 4)
        self.assertEqual(optimizer.health, [1.5] * 4)
        self.assertEqual(optimizer.chemotactic_step, 0)
        self.assertEqual(optimizer.reproduction_step, 1)
        self.assertEqual(optimizer.elimination_dispersal_step, 0)

    def test_reproduction_and_elimination_dispersal_preserve_population_size(self):
        optimizer = self.make_optimizer(
            chemotactic_steps=1,
            reproduction_steps=1,
            elimination_dispersal_probability=1.0,
        )
        optimizer.init()

        optimizer.main_loop_iteration()

        self.assertEqual(len(optimizer.current_population), 4)
        self.assertEqual(optimizer.population_size, 4)
        self.assertEqual(optimizer.elimination_dispersal_step, 1)
        self.assertEqual(optimizer.health, [0.0] * 4)
        self.assertEqual(optimizer.step_sizes, [0.5] * 4)
        self.assertEqual(optimizer.evaluation, 16)
        self.assertTrue(optimizer.should_finish())

    def test_optimize_runs_until_natural_elimination_termination(self):
        optimizer = self.make_optimizer()

        result = optimizer.optimize()

        self.assertIs(result, optimizer.best_solution)
        self.assertEqual(optimizer.iteration, 4)
        self.assertEqual(optimizer.elimination_dispersal_step, 1)
        self.assertEqual(len(optimizer.current_population), 4)
        self.assertEqual(optimizer.step_sizes, [0.5] * 4)
        self.assertIsNotNone(optimizer.execution_started)
        self.assertIsNotNone(optimizer.execution_ended)

    def test_copy_preserves_configured_strategies_and_runtime_state(self):
        optimizer = self.make_optimizer(chemotactic_steps=1)
        optimizer.init()
        optimizer.main_loop_iteration()

        copied = optimizer.copy()

        self.assertIsNot(optimizer, copied)
        self.assertIsInstance(copied.step_size_support, BfoStepSizeSupportFixed)
        self.assertIsInstance(copied.swarming_support, BfoSwarmingSupportIdle)
        self.assertEqual(copied.step_sizes, optimizer.step_sizes)
        self.assertEqual(copied.health, optimizer.health)
        self.assertEqual(len(copied.current_population), len(optimizer.current_population))
        self.assertIsNot(copied.current_population, optimizer.current_population)
        self.assertEqual(copied.execution_started, optimizer.execution_started)
        self.assertEqual(copied.execution_ended, optimizer.execution_ended)
        self.assertEqual(copied.time_when_best_found, optimizer.time_when_best_found)
        self.assertIsNot(copied.best_solution, optimizer.best_solution)

    def test_construction_parameters_create_equivalent_optimizer(self):
        optimizer = self.make_optimizer()
        parameters = BfoOptimizerConstructionParameters(
            movement_support=DeterministicMovement(),
            swarming_support=BfoSwarmingSupportIdle(),
            step_size_support=BfoStepSizeSupportFixed(0.5),
            population_size=4,
            chemotactic_steps=2,
            swim_length=1,
            reproduction_steps=2,
            elimination_dispersal_events=1,
            elimination_dispersal_probability=1.0,
            finish_control=FinishControl(criteria=""),
            problem=ProblemVoidMinSO("linear"),
            solution_template=LinearSolution(),
            random_seed=7,
        )

        constructed = BfoOptimizer.from_construction_tuple(parameters)

        self.assertEqual(constructed.population_size, optimizer.population_size)
        self.assertEqual(constructed.chemotactic_steps, optimizer.chemotactic_steps)
        self.assertEqual(constructed.swim_length, optimizer.swim_length)
        self.assertIsInstance(constructed.step_size_support, BfoStepSizeSupportFixed)
        self.assertIsInstance(constructed.swarming_support, BfoSwarmingSupportIdle)

    def test_constructor_rejects_invalid_population_configuration(self):
        with self.assertRaises(ValueError):
            self.make_optimizer(population_size=3)
        with self.assertRaises(ValueError):
            self.make_optimizer(population_size=1)

    def test_constructor_rejects_invalid_execution_parameters(self):
        for name, value in (
            ("chemotactic_steps", 0),
            ("swim_length", -1),
            ("reproduction_steps", 0),
            ("elimination_dispersal_events", 0),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.make_optimizer(**{name: value})

        with self.assertRaises(ValueError):
            self.make_optimizer(elimination_dispersal_probability=1.1)


if __name__ == "__main__":
    unittest.main()
