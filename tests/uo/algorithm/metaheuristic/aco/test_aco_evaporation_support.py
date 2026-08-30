import unittest

from uo.algorithm.metaheuristic.aco.aco_evaporation_support import (
    AcoEvaporationSupportFixed,
    AcoEvaporationSupportAdaptive,
)
from uo.algorithm.metaheuristic.finish_control import FinishControl


class FakeSolution:
    def __init__(self, tour, cost):
        self.representation = type("Repr", (), {"tour": tour})()
        self._cost = cost

    def tour_cost(self, problem, tour):
        return self._cost


class FakeOptimizer:
    def __init__(self, n, rho, tau_max, tau_min, iteration=0, iterations_max=10, stagnation_count=0, stagnation_limit=100):
        self.problem = type("Problem", (), {"n": n})()
        self.rho = rho
        self.tau = [[1.0] * n for _ in range(n)]
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.iteration = iteration
        self.finish_control = FinishControl(criteria="iterations", iterations_max=iterations_max)
        self.stagnation_count = stagnation_count
        self.stagnation_limit = stagnation_limit


class TestAcoEvaporationSupportFixed(unittest.TestCase):

    def test_apply_evaporates_all_edges(self):
        optimizer = FakeOptimizer(n=3, rho=0.1, tau_max=2.0, tau_min=0.5)
        iteration_best = FakeSolution([0, 1, 2], cost=3.0)
        global_best = FakeSolution([0, 1, 2], cost=3.0)
        support = AcoEvaporationSupportFixed()

        support.apply(optimizer, iteration_best, global_best)

        for row in optimizer.tau:
            for value in row:
                self.assertLessEqual(value, optimizer.tau_max)
                self.assertGreaterEqual(value, optimizer.tau_min)

    def test_apply_deposits_from_iteration_best_before_halfway(self):
        optimizer = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, iteration=1, iterations_max=10)
        iteration_best = FakeSolution([0, 1, 2], cost=3.0)
        global_best = FakeSolution([0, 2, 1], cost=10.0)
        support = AcoEvaporationSupportFixed()

        support.apply(optimizer, iteration_best, global_best)

        self.assertGreater(optimizer.tau[0][1], optimizer.tau[0][2])

    def test_apply_deposits_from_global_best_after_halfway(self):
        optimizer = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, iteration=8, iterations_max=10)
        iteration_best = FakeSolution([0, 1, 2], cost=3.0)
        global_best = FakeSolution([0, 2, 1], cost=10.0)
        support = AcoEvaporationSupportFixed()

        support.apply(optimizer, iteration_best, global_best)

        self.assertGreater(optimizer.tau[0][2], optimizer.tau[0][1])

    def test_copy_returns_new_instance(self):
        support = AcoEvaporationSupportFixed()

        copied = support.copy()

        self.assertIsNot(support, copied)
        self.assertIsInstance(copied, AcoEvaporationSupportFixed)


class TestAcoEvaporationSupportAdaptive(unittest.TestCase):

    def test_apply_evaporates_faster_when_stagnant(self):
        fresh = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, stagnation_count=0, stagnation_limit=100)
        stuck = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, stagnation_count=100, stagnation_limit=100)
        solution = FakeSolution([0, 1, 2], cost=3.0)
        support = AcoEvaporationSupportAdaptive(rho_scale=2.0)

        fresh.tau = [[2.0] * 3 for _ in range(3)]
        stuck.tau = [[2.0] * 3 for _ in range(3)]
        support.apply(fresh, solution, solution)
        support.apply(stuck, solution, solution)

        self.assertLess(stuck.tau[1][2], fresh.tau[1][2])

    def test_apply_deposits_from_global_best_when_heavily_stagnant(self):
        optimizer = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, stagnation_count=90, stagnation_limit=100)
        iteration_best = FakeSolution([0, 1, 2], cost=3.0)
        global_best = FakeSolution([0, 2, 1], cost=10.0)
        support = AcoEvaporationSupportAdaptive(rho_scale=2.0)

        support.apply(optimizer, iteration_best, global_best)

        self.assertGreater(optimizer.tau[0][2], optimizer.tau[0][1])

    def test_apply_deposits_from_iteration_best_when_fresh(self):
        optimizer = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, stagnation_count=0, stagnation_limit=100)
        iteration_best = FakeSolution([0, 1, 2], cost=3.0)
        global_best = FakeSolution([0, 2, 1], cost=10.0)
        support = AcoEvaporationSupportAdaptive(rho_scale=2.0)

        support.apply(optimizer, iteration_best, global_best)

        self.assertGreater(optimizer.tau[0][1], optimizer.tau[0][2])

    def test_copy_preserves_rho_scale_behavior(self):
        optimizer_a = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, stagnation_count=100, stagnation_limit=100)
        optimizer_b = FakeOptimizer(n=3, rho=0.1, tau_max=5.0, tau_min=0.1, stagnation_count=100, stagnation_limit=100)
        optimizer_a.tau = [[2.0] * 3 for _ in range(3)]
        optimizer_b.tau = [[2.0] * 3 for _ in range(3)]
        solution = FakeSolution([0, 1, 2], cost=3.0)

        support = AcoEvaporationSupportAdaptive(rho_scale=3.5)
        copied = support.copy()

        self.assertIsNot(support, copied)
        support.apply(optimizer_a, solution, solution)
        copied.apply(optimizer_b, solution, solution)
        self.assertEqual(optimizer_a.tau[1][2], optimizer_b.tau[1][2])


if __name__ == "__main__":
    unittest.main()
