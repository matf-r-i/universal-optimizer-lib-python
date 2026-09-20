import math
import unittest
from random import Random

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_movement_support import (
    BfoMovementSupportReal,
)


class VectorSolution:
    def __init__(self, representation):
        self.representation = representation

    def copy(self):
        return VectorSolution(self.representation.copy())

    def init_from(self, representation, problem):
        self.representation = representation


class TestBfoMovementSupportReal(unittest.TestCase):

    def test_direction_is_normalized_and_move_clips_without_mutating_parent(self):
        support = BfoMovementSupportReal([-1.0, -1.0], [1.0, 1.0])
        solution = VectorSolution([0.25, -0.25])

        direction = support.tumble_direction(solution, Random(3))
        equal_seed_direction = support.tumble_direction(solution, Random(3))
        candidate = support.move(solution, (1.0, -1.0), 2.0, object())

        self.assertAlmostEqual(math.sqrt(sum(value * value for value in direction)), 1.0)
        self.assertEqual(direction, equal_seed_direction)
        self.assertEqual(solution.representation, [0.25, -0.25])
        self.assertEqual(candidate.representation, [1.0, -1.0])
        self.assertIsNot(candidate, solution)

    def test_invalid_bounds_dimensions_and_step_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            BfoMovementSupportReal([], [])
        with self.assertRaises(ValueError):
            BfoMovementSupportReal([0.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            BfoMovementSupportReal([2.0], [1.0])

        support = BfoMovementSupportReal([0.0, 0.0], [1.0, 1.0])
        solution = VectorSolution([0.5, 0.5])
        with self.assertRaises(ValueError):
            support.move(solution, (1.0,), 0.1, object())
        with self.assertRaises(TypeError):
            support.move(solution, (1.0, 1.0), True, object())


if __name__ == "__main__":
    unittest.main()
