import unittest
from random import Random

from uo.problem.problem import Problem
from uo.solution.quality_of_solution import QualityOfSolution
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.finish_control import FinishControl

from uo.algorithm.metaheuristic.aco.aco_optimizer import AcoOptimizerConstructionParameters
from uo.algorithm.metaheuristic.aco.aco_optimizer_standard import AcoOptimizerStandard
from uo.algorithm.metaheuristic.aco.aco_construction_support import AcoConstructionSupport
from uo.algorithm.metaheuristic.aco.aco_evaporation_support import AcoEvaporationSupportFixed


class TourProblem(Problem):
    def __init__(self, distances):
        super().__init__("tour_problem", False, False)
        self.distances = distances
        self.n = len(distances)

    def copy(self):
        return TourProblem(self.distances)

    def __str__(self):
        return self.string_rep("|")

    def __repr__(self):
        return self.string_rep("\n")

    def __format__(self, spec):
        return self.string_rep("|")


class TourRepresentation:
    def __init__(self, tour):
        self.tour = tour

    def copy(self):
        return TourRepresentation(self.tour.copy())


class TourSolution(Solution[TourRepresentation, str]):
    def __init__(self, random_seed=None):
        super().__init__(random_seed, None, None, None, None, False)
        self.is_minimization = False

    def copy(self):
        sol = TourSolution(self.random_seed)
        sol.copy_from(self)
        return sol

    def copy_from(self, original):
        super().copy_from(original)

    def argument(self, representation):
        return str(representation.tour)

    def init_random(self, problem):
        tour = list(range(problem.n))
        Random(self.random_seed).shuffle(tour)
        self.representation = TourRepresentation(tour)

    def init_from(self, representation, problem):
        self.representation = representation.copy()

    def native_representation(self, representation_str):
        values = representation_str.strip("[]").split(",")
        return TourRepresentation([int(x) for x in values])

    def tour_cost(self, problem, tour):
        n = len(tour)
        return sum(problem.distances[tour[k]][tour[(k + 1) % n]] for k in range(n))

    def calculate_quality_directly(self, representation, problem):
        cost = self.tour_cost(problem, representation.tour)
        return QualityOfSolution(
            objective_value=-cost,
            objective_values=None,
            fitness_value=-cost,
            fitness_values=None,
            is_feasible=True,
        )

    def representation_distance_directly(self, representation_1, representation_2):
        return float(sum(1 for a, b in zip(representation_1.tour, representation_2.tour) if a != b))

    def __str__(self):
        return self.string_rep("|")

    def __repr__(self):
        return self.string_rep("\n")

    def __format__(self, spec):
        return self.string_rep("|")


class WeightedChoiceConstructionSupport(AcoConstructionSupport):
    def copy(self):
        return WeightedChoiceConstructionSupport()

    def construct(self, problem, solution, optimizer):
        from random import choices
        n = problem.n
        tour = [0]
        visited = [False] * n
        visited[0] = True
        for _ in range(n - 1):
            current = tour[-1]
            candidates = [j for j in range(n) if not visited[j]]
            weights = [
                (optimizer.tau[current][j] ** optimizer.alpha) * (optimizer.eta[current][j] ** optimizer.beta)
                for j in candidates
            ]
            next_city = choices(candidates, weights=weights, k=1)[0]
            tour.append(next_city)
            visited[next_city] = True
        solution.representation = TourRepresentation(tour)

    def local_search(self, problem, solution, optimizer):
        pass


# n=7 keeps the MMAS tau_min formula valid (p_best**(1/n) >= 2/n)
CIRCLE_DISTANCES = [
    [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
    [1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0],
    [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
    [4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
    [3.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0],
    [2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0],
]


class TestAcoOptimizerStandard(unittest.TestCase):

    def make_optimizer(self, seed=11, iterations_max=10):
        params = AcoOptimizerConstructionParameters()
        params.construction_support = WeightedChoiceConstructionSupport()
        params.evaporation_support = AcoEvaporationSupportFixed()
        params.n_ants = 7
        params.alpha = 1.0
        params.beta = 2.0
        params.rho = 0.1
        params.p_best = 0.05
        params.stagnation_limit = 100
        params.finish_control = FinishControl(criteria="iterations", iterations_max=iterations_max)
        params.problem = TourProblem(CIRCLE_DISTANCES)
        params.solution_template = TourSolution()
        params.random_seed = seed
        return AcoOptimizerStandard.from_construction_tuple(params)

    def test_optimize_respects_iterations_max(self):
        optimizer = self.make_optimizer(iterations_max=15)

        optimizer.optimize()

        self.assertEqual(optimizer.iteration, 15)

    def test_optimize_finds_a_feasible_tour_of_correct_length(self):
        optimizer = self.make_optimizer()

        best = optimizer.optimize()

        self.assertEqual(sorted(best.representation.tour), list(range(7)))
        self.assertTrue(best.is_feasible)

    def test_tau_matrix_stays_within_bounds_after_optimize(self):
        optimizer = self.make_optimizer()

        optimizer.optimize()

        for row in optimizer.tau:
            for value in row:
                self.assertGreaterEqual(value, optimizer.tau_min)
                self.assertLessEqual(value, optimizer.tau_max)

    def test_copy_returns_independent_optimizer(self):
        optimizer = self.make_optimizer()

        copied = optimizer.copy()

        self.assertIsNot(optimizer, copied)
        self.assertEqual(copied.n_ants, optimizer.n_ants)
        self.assertEqual(copied.alpha, optimizer.alpha)
        self.assertEqual(copied.stagnation_limit, optimizer.stagnation_limit)


if __name__ == "__main__":
    unittest.main()
