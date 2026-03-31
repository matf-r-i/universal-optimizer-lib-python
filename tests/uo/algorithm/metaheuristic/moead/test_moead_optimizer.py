import unittest
import numpy as np
from datetime import datetime

from uo.algorithm.metaheuristic.additional_statistics_control import AdditionalStatisticsControl
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.moead.moead_optimizer import MoeadOptimizer
from uo.algorithm.metaheuristic.moead.moead_variation_support_binary import MoeadVariationSupportBinary
from uo.algorithm.output_control import OutputControl
from uo.problem.problem import Problem
from uo.solution.solution import Solution


class DummyProblem(Problem):
    def __init__(self):
        super().__init__(
            is_minimization=True,
            is_multi_objective=True,
            name="dummy_problem"
        )

    def copy(self):
        return DummyProblem()

    def __str__(self):
        return "DummyProblem"

    def __repr__(self):
        return "DummyProblem"

    def __format__(self, spec):
        return "DummyProblem"


class DummySolution(Solution):
    def __init__(self, representation=None):
        super().__init__(
            random_seed=None,
            fitness_value=None,
            fitness_values=None,
            objective_value=None,
            objective_values=None,
            is_feasible=True
        )
        self.representation = representation
        self.objective_values = None
        self.fitness_values = None
        self.objective_value = None
        self.fitness_value = None

    def copy(self):
        new_sol = DummySolution(
            None if self.representation is None else np.array(self.representation, copy=True)
        )
        if self.objective_values is not None:
            new_sol.objective_values = list(self.objective_values)
        if self.fitness_values is not None:
            new_sol.fitness_values = list(self.fitness_values)
        new_sol.objective_value = self.objective_value
        new_sol.fitness_value = self.fitness_value
        return new_sol

    @property
    def argument(self):
        return self.representation

    @property
    def native_representation(self):
        return self.representation

    def init_random(self, problem):
        rng = np.random.default_rng()
        self.representation = rng.integers(0, 2, size=6, dtype=int)

    def init_from(self, solution):
        self.representation = None if solution.representation is None else np.array(solution.representation, copy=True)

    def copy_from(self, solution):
        self.representation = None if solution.representation is None else np.array(solution.representation, copy=True)
        self.objective_values = None if solution.objective_values is None else list(solution.objective_values)
        self.fitness_values = None if solution.fitness_values is None else list(solution.fitness_values)
        self.objective_value = solution.objective_value
        self.fitness_value = solution.fitness_value

    def calculate_quality_directly(self, problem):
        return float(np.sum(self.representation))

    def representation_distance_directly(self, other):
        if self.representation is None or other.representation is None:
            return 0.0
        return float(np.sum(self.representation != other.representation))

    def evaluate(self, problem):
        ones = int(np.sum(self.representation))
        zeros = int(len(self.representation) - ones)
        self.objective_values = [float(ones), float(zeros)]
        self.fitness_values = [float(ones), float(zeros)]
        self.objective_value = float(ones)
        self.fitness_value = float(ones)

    def is_better(self, solution):
        return False

    def __str__(self):
        return "DummySolution"

    def __repr__(self):
        return "DummySolution"

    def __format__(self, spec):
        return "DummySolution"


class TestMoeadOptimizer(unittest.TestCase):

    def build_optimizer(self):
        finish_control = FinishControl(
            criteria='evaluations & iterations',
            evaluations_max=100,
            iterations_max=10,
            seconds_max=0
        )

        optimizer = MoeadOptimizer(
            moead_variation_support=MoeadVariationSupportBinary(
                crossover_probability=1.0,
                mutation_probability=0.0
            ),
            population_size=6,
            neighborhood_size=3,
            max_number_of_replaced_neighbors=2,
            finish_control=finish_control,
            problem=DummyProblem(),
            solution_template=DummySolution(),
            output_control=None,
            random_seed=42,
            additional_statistics_control=None,
            lattice_parameter_H=None
        )
        return optimizer

    def test_init(self):
        optimizer = self.build_optimizer()
        optimizer.execution_started = datetime.now()
        optimizer.init()

        self.assertEqual(len(optimizer.current_population), 6)
        self.assertIsNotNone(optimizer.weight_setup)
        self.assertIsNotNone(optimizer.ideal_point)
        self.assertIsNotNone(optimizer.best_solution)
        self.assertGreaterEqual(optimizer.evaluation, 6)

    def test_main_loop_iteration(self):
        optimizer = self.build_optimizer()
        optimizer.execution_started = datetime.now()
        optimizer.init()

        prev_eval = optimizer.evaluation
        prev_iter = optimizer.iteration

        optimizer.main_loop_iteration()

        self.assertGreater(optimizer.evaluation, prev_eval)
        self.assertEqual(optimizer.iteration, prev_iter + 1)
        self.assertIsNotNone(optimizer.best_solution)
        self.assertTrue(len(optimizer.nondominated_archive) >= 1)

    def test_copy(self):
        optimizer = self.build_optimizer()
        optimizer.execution_started = datetime.now()
        optimizer.init()

        copied = optimizer.copy()

        self.assertIsInstance(copied, MoeadOptimizer)
        self.assertEqual(copied.population_size, optimizer.population_size)
        self.assertEqual(len(copied.current_population), len(optimizer.current_population))

    def test_constructor_rejects_single_objective_problem(self):
        class SingleObjectiveProblem(DummyProblem):
            def __init__(self):
                super().__init__()
                self.is_multi_objective = False

        finish_control = FinishControl(
            criteria='evaluations & iterations',
            evaluations_max=100,
            iterations_max=10,
            seconds_max=0
        )

        with self.assertRaises(ValueError):
            MoeadOptimizer(
                moead_variation_support=MoeadVariationSupportBinary(),
                population_size=6,
                neighborhood_size=3,
                max_number_of_replaced_neighbors=2,
                finish_control=finish_control,
                problem=SingleObjectiveProblem(),
                solution_template=DummySolution(),
                output_control=None,
                random_seed=42,
                additional_statistics_control=None,
                lattice_parameter_H=None
            )


if __name__ == "__main__":
    unittest.main()