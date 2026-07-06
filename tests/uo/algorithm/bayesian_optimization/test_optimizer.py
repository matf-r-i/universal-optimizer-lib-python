import unittest

import numpy as np

from uo.algorithm.algorithm import Algorithm
from uo.algorithm.bayesian_optimization.optimizer import (
    AcquisitionConfig,
    BayesianOptimizer,
    BayesianOptimizerConstructionParameters,
)
from uo.problem.problem import Problem
from uo.solution.quality_of_solution import QualityOfSolution
from uo.solution.solution import Solution


class VectorProblem(Problem):
    def __init__(self, center, is_minimization=True):
        super().__init__("vector_problem", is_minimization, False)
        self.center = np.asarray(center, dtype=np.float64)

    def copy(self):
        return VectorProblem(self.center.copy(), self.is_minimization)

    def __str__(self):
        return self.string_rep("|")

    def __repr__(self):
        return self.string_rep("\n")

    def __format__(self, spec):
        return self.string_rep("|")


class VectorSolution(Solution[np.ndarray, np.ndarray]):
    def __init__(self, feasible=True):
        super().__init__(None, None, None, None, None, feasible)

    def copy(self):
        result = VectorSolution(self.is_feasible)
        result.copy_from(self)
        return result

    def copy_from(self, original):
        super().copy_from(original)

    def argument(self, representation):
        return representation

    def init_random(self, problem):
        raise NotImplementedError

    def init_from(self, representation, problem):
        self.representation = np.asarray(representation, dtype=np.float64).copy()

    def native_representation(self, representation_str):
        return np.fromstring(representation_str.strip("[]"), sep=" ")

    def calculate_quality_directly(self, representation, problem):
        objective = float(np.sum((representation - problem.center) ** 2))
        fitness = -objective if problem.is_minimization else objective
        return QualityOfSolution(
            objective_value=objective,
            objective_values=None,
            fitness_value=fitness,
            fitness_values=None,
            is_feasible=self.is_feasible,
        )

    def representation_distance_directly(self, representation_1, representation_2):
        return float(np.linalg.norm(representation_1 - representation_2))

    def __str__(self):
        return self.string_rep("|")

    def __repr__(self):
        return self.string_rep("\n")

    def __format__(self, spec):
        return self.string_rep("|")


class TestBayesianOptimizer(unittest.TestCase):
    def make_optimizer(self, seed=11, budget=12):
        return BayesianOptimizer(
            problem=VectorProblem([0.25, 0.25]),
            solution_template=VectorSolution(),
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            evaluation_budget=budget,
            number_of_initial_points=5,
            random_seed=seed,
            acquisition_config=AcquisitionConfig(number_of_restarts=3),
        )

    def test_optimizer_derives_directly_from_algorithm_contract(self):
        optimizer = self.make_optimizer()
        self.assertIsInstance(optimizer, Algorithm)
        self.assertNotIn("Metaheuristic", [base.__name__ for base in type(optimizer).__mro__])

    def test_optimizer_respects_budget_bounds_and_history_shapes(self):
        optimizer = self.make_optimizer()
        best = optimizer.optimize()
        self.assertEqual(optimizer.evaluation, 12)
        self.assertEqual(optimizer.iteration, 7)
        self.assertEqual(optimizer.x_history.shape, (12, 2))
        self.assertEqual(optimizer.target_history.shape, (12,))
        self.assertEqual(optimizer.acquisition_history.shape, (7,))
        self.assertTrue(np.all(optimizer.x_history >= 0.0))
        self.assertTrue(np.all(optimizer.x_history <= 1.0))
        self.assertAlmostEqual(best.fitness_value, -float(np.min(optimizer.target_history)))
        self.assertLess(best.objective_value, 0.02)

    def test_seed_produces_reproducible_history(self):
        first = self.make_optimizer(seed=4)
        second = self.make_optimizer(seed=4)
        first.optimize()
        second.optimize()
        self.assertTrue(np.allclose(first.x_history, second.x_history))
        self.assertTrue(np.allclose(first.target_history, second.target_history))

    def test_histories_and_bounds_are_returned_as_copies(self):
        optimizer = self.make_optimizer(budget=5)
        optimizer.optimize()
        history = optimizer.x_history
        bounds = optimizer.bounds
        history[:] = 99.0
        bounds[:] = 99.0
        self.assertFalse(np.all(optimizer.x_history == 99.0))
        self.assertFalse(np.all(optimizer.bounds == 99.0))

    def test_construction_parameters_and_copy(self):
        parameters = BayesianOptimizerConstructionParameters(
            problem=VectorProblem([0.5]),
            solution_template=VectorSolution(),
            bounds=[(0.0, 1.0)],
            evaluation_budget=5,
            random_seed=2,
        )
        optimizer = BayesianOptimizer.from_construction_tuple(parameters)
        copied = optimizer.copy()
        self.assertIsInstance(optimizer, BayesianOptimizer)
        self.assertIsNot(optimizer, copied)
        self.assertEqual(copied.evaluation_budget, 5)

    def test_maximization_problem_uses_fitness_consistently(self):
        optimizer = BayesianOptimizer(
            problem=VectorProblem([0.5], is_minimization=False),
            solution_template=VectorSolution(),
            bounds=[(0.0, 1.0)],
            evaluation_budget=8,
            number_of_initial_points=4,
            random_seed=8,
            acquisition_config=AcquisitionConfig(number_of_restarts=2),
        )
        best = optimizer.optimize()
        self.assertAlmostEqual(best.fitness_value, -float(np.min(optimizer.target_history)))
        self.assertEqual(best.fitness_value, max(-optimizer.target_history))

    def test_invalid_optimizer_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            BayesianOptimizer(VectorProblem([0.0]), VectorSolution(), [(0.0, 1.0)], 0)
        with self.assertRaises(ValueError):
            BayesianOptimizer(VectorProblem([0.0]), VectorSolution(), [(1.0, 0.0)], 5)
        multi_objective = VectorProblem([0.0])
        multi_objective.is_multi_objective = True
        with self.assertRaises(ValueError):
            BayesianOptimizer(multi_objective, VectorSolution(), [(0.0, 1.0)], 5)

    def test_infeasible_solution_is_rejected(self):
        optimizer = BayesianOptimizer(
            VectorProblem([0.0]),
            VectorSolution(feasible=False),
            [(0.0, 1.0)],
            1,
        )
        with self.assertRaises(ValueError):
            optimizer.optimize()


if __name__ == "__main__":
    unittest.main()
