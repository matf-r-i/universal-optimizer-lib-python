import unittest
import numpy as np

from uo.algorithm.metaheuristic.moead.moead_variation_support_binary import MoeadVariationSupportBinary
from uo.problem.problem import Problem
from uo.solution.solution import Solution


class DummyProblem(Problem):
    def __init__(self):
        super().__init__(is_minimization=True, is_multi_objective=True, name="dummy_problem")

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
        self.representation = np.array([0, 1, 0, 1], dtype=int)

    def init_from(self, solution):
        self.representation = None if solution.representation is None else np.array(solution.representation, copy=True)

    def copy_from(self, solution):
        self.representation = None if solution.representation is None else np.array(solution.representation, copy=True)
        self.objective_values = None if solution.objective_values is None else list(solution.objective_values)
        self.fitness_values = None if solution.fitness_values is None else list(solution.fitness_values)    
        self.objective_value = solution.objective_value
        self.fitness_value = solution.fitness_value

    def calculate_quality_directly(self, problem):
        s = int(np.sum(self.representation))
        return float(s)

    def representation_distance_directly(self, other):
        if self.representation is None or other.representation is None:
            return 0.0
        return float(np.sum(self.representation != other.representation))

    def evaluate(self, problem):
        s = int(np.sum(self.representation))
        self.objective_values = [float(s), float(len(self.representation) - s)]
        self.fitness_values = [float(s), float(len(self.representation) - s)]
        self.objective_value = float(s)
        self.fitness_value = float(s)

    def is_better(self, solution):
        return False

    def __str__(self):
        return "DummySolution"

    def __repr__(self):
        return "DummySolution"

    def __format__(self, spec):
        return "DummySolution"
    

class DummyOptimizer:
    def __init__(self):
        self.random_seed = 42
        self.evaluation = 0

    def write_output_values_if_needed(self, *args, **kwargs):
        pass


class TestMoeadVariationSupportBinary(unittest.TestCase):

    def test_copy(self):
        support = MoeadVariationSupportBinary(crossover_probability=0.9, mutation_probability=0.1)
        copied = support.copy()

        self.assertIsInstance(copied, MoeadVariationSupportBinary)
        self.assertEqual(copied.crossover_probability, 0.9)
        self.assertEqual(copied.mutation_probability, 0.1)

    def test_generate_offspring(self):
        support = MoeadVariationSupportBinary(crossover_probability=1.0, mutation_probability=0.0)

        problem = DummyProblem()
        parent1 = DummySolution(np.array([0, 0, 0, 0], dtype=int))
        parent2 = DummySolution(np.array([1, 1, 1, 1], dtype=int))
        child = DummySolution()
        optimizer = DummyOptimizer()

        support.generate_offspring(
            problem=problem,
            parent1=parent1,
            parent2=parent2,
            child=child,
            optimizer=optimizer
        )

        self.assertIsNotNone(child.representation)
        self.assertEqual(len(child.representation), 4)
        self.assertIsNotNone(child.objective_values)
        self.assertEqual(optimizer.evaluation, 1)

    def test_generate_offspring_with_none_representation(self):
        support = MoeadVariationSupportBinary()

        problem = DummyProblem()
        parent1 = DummySolution(None)
        parent2 = DummySolution(np.array([1, 1, 1], dtype=int))
        child = DummySolution()
        optimizer = DummyOptimizer()

        support.generate_offspring(
            problem=problem,
            parent1=parent1,
            parent2=parent2,
            child=child,
            optimizer=optimizer
        )

        self.assertIsNone(child.representation)
        self.assertEqual(optimizer.evaluation, 0)


if __name__ == "__main__":
    unittest.main()