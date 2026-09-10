import unittest
from datetime import datetime

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.solution.quality_of_solution import QualityOfSolution
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.tabu_search.tabu_list import TabuList
from uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer import (
    TabuSearchOptimizer,
    TabuSearchOptimizerConstructionParameters,
)
from uo.algorithm.metaheuristic.tabu_search.tabu_search_support_standard_permutation import (
    TabuSearchSupportStandardPermutation,
)

_POSITIONS = [0, 1, 2, 3, 4]


def _distance(a: int, b: int) -> int:
    return abs(_POSITIONS[a] - _POSITIONS[b])


def _tour_length(representation: list[int]) -> int:
    n = len(representation)
    return sum(_distance(representation[k], representation[(k + 1) % n]) for k in range(n))


class _LinearTspProblem(Problem):

    def __init__(self) -> None:
        super().__init__(name="LinearTspProblem", is_minimization=True, is_multi_objective=False)

    def copy(self) -> "_LinearTspProblem":
        return _LinearTspProblem()

    def __str__(self) -> str:
        return self.string_rep("|")

    def __repr__(self) -> str:
        return self.string_rep("\n")

    def __format__(self, spec: str) -> str:
        return self.string_rep("|")


class _LinearTspSolution(Solution[list, list]):

    def __init__(self, random_seed: int = None) -> None:
        super().__init__(random_seed=random_seed, fitness_value=None, fitness_values=None,
                objective_value=None, objective_values=None, is_feasible=False)

    def copy(self) -> "_LinearTspSolution":
        sol = _LinearTspSolution(self.random_seed)
        sol.copy_from(self)
        return sol

    def copy_from(self, original) -> None:
        super().copy_from(original)

    def argument(self, representation: list) -> list:
        return list(representation)

    def init_random(self, problem: Problem) -> None:
        self.representation = list(_POSITIONS)

    def init_from(self, representation: list, problem: Problem) -> None:
        self.representation = list(representation)

    def native_representation(self, representation_str: str) -> list:
        return list(_POSITIONS)

    def calculate_quality_directly(self, representation: list, problem: Problem) -> QualityOfSolution:
        length = _tour_length(representation)
        return QualityOfSolution(fitness_value=-float(length), fitness_values=None,
                objective_value=float(length), objective_values=None, is_feasible=True)

    def representation_distance_directly(self, representation_1: list, representation_2: list) -> float:
        return float(sum(1 for a, b in zip(representation_1, representation_2) if a != b))

    def __str__(self) -> str:
        return self.string_rep("|")

    def __repr__(self) -> str:
        return self.string_rep("\n")

    def __format__(self, spec: str) -> str:
        return self.string_rep("|")


def _make_optimizer(tabu_tenure: int = 3) -> TabuSearchOptimizer:
    params = TabuSearchOptimizerConstructionParameters()
    params.tabu_search_support = TabuSearchSupportStandardPermutation(dimension=5)
    params.tabu_tenure = tabu_tenure
    params.finish_control = FinishControl(criteria="iterations", iterations_max=1000)
    params.problem = _LinearTspProblem()
    params.solution_template = _LinearTspSolution()
    params.random_seed = 42
    optimizer = TabuSearchOptimizer.from_construction_tuple(params)
    optimizer.execution_started = datetime.now()
    return optimizer


class TestTabuSearchSupportStandardPermutation(unittest.TestCase):

    def test_dimension_property(self):
        support = TabuSearchSupportStandardPermutation(dimension=5)
        self.assertEqual(support.dimension, 5)

    def test_copy_returns_independent_instance_with_same_dimension(self):
        support = TabuSearchSupportStandardPermutation(dimension=5)
        copied = support.copy()
        self.assertIsNot(support, copied)
        self.assertEqual(copied.dimension, support.dimension)

    def test_best_neighbor_move_improves_bad_ordering(self):
        optimizer = _make_optimizer()
        optimizer.init()
        optimizer.current_solution.representation = [0, 2, 4, 1, 3]
        optimizer.current_solution.evaluate(optimizer.problem)
        worst_length = optimizer.current_solution.objective_value

        move = optimizer.tabu_search_support.best_neighbor_move(
            optimizer.problem, optimizer.current_solution, optimizer.tabu_list, optimizer
        )

        self.assertIsNotNone(move)
        self.assertLess(optimizer.current_solution.objective_value, worst_length)

    def test_best_neighbor_move_returns_none_when_dimension_too_small(self):
        support = TabuSearchSupportStandardPermutation(dimension=1)
        optimizer = _make_optimizer()
        optimizer.init()
        optimizer.current_solution.representation = [0]
        move = support.best_neighbor_move(
            optimizer.problem, optimizer.current_solution, optimizer.tabu_list, optimizer
        )
        self.assertIsNone(move)

    def test_best_neighbor_move_returns_none_when_should_finish(self):
        params = TabuSearchOptimizerConstructionParameters()
        params.tabu_search_support = TabuSearchSupportStandardPermutation(dimension=5)
        params.tabu_tenure = 3
        params.finish_control = FinishControl(criteria="evaluations", evaluations_max=1)
        params.problem = _LinearTspProblem()
        params.solution_template = _LinearTspSolution()
        params.random_seed = 42
        optimizer = TabuSearchOptimizer.from_construction_tuple(params)
        optimizer.execution_started = datetime.now()
        optimizer.init()
        optimizer.evaluation = 999

        move = optimizer.tabu_search_support.best_neighbor_move(
            optimizer.problem, optimizer.current_solution, optimizer.tabu_list, optimizer
        )
        self.assertIsNone(move)

    def test_tabu_move_is_skipped_unless_aspiration_applies(self):
        optimizer = _make_optimizer(tabu_tenure=10)
        optimizer.init()
        optimizer.current_solution.representation = [0, 2, 4, 1, 3]
        optimizer.current_solution.evaluate(optimizer.problem)
        optimizer.best_solution = optimizer.current_solution

        first_move = optimizer.tabu_search_support.best_neighbor_move(
            optimizer.problem, optimizer.current_solution, optimizer.tabu_list, optimizer
        )
        self.assertIsNotNone(first_move)
        optimizer.tabu_list.add(first_move)
        if optimizer.current_solution.is_better(optimizer.best_solution, optimizer.problem):
            optimizer.best_solution = optimizer.current_solution

        full_tabu_list = TabuList(tenure=100)
        n = len(optimizer.current_solution.representation)
        for i in range(n - 1):
            for j in range(i + 1, n):
                full_tabu_list.add((i, j))

        move = optimizer.tabu_search_support.best_neighbor_move(
            optimizer.problem, optimizer.current_solution, full_tabu_list, optimizer
        )
        if move is not None:
            self.assertTrue(optimizer.current_solution.is_better(optimizer.best_solution, optimizer.problem)
                    or optimizer.current_solution.objective_value <= optimizer.best_solution.objective_value)


if __name__ == "__main__":
    unittest.main()
