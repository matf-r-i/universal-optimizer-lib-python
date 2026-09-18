import math
import unittest

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_swarming_support import (
    BfoSwarmingSupportReal,
)


class DistanceSolution:
    def __init__(self, representation):
        self.representation = representation

    def representation_distance(self, first, second):
        return abs(first - second)


class TestBfoSwarmingSupportReal(unittest.TestCase):

    def test_interaction_uses_distance_based_attraction_and_repulsion(self):
        support = BfoSwarmingSupportReal(2.0, 1.0, 1.0, 0.5)
        bacterium = DistanceSolution(0.0)
        other = DistanceSolution(2.0)

        actual = support.interaction_value(bacterium, [bacterium, other])
        expected = (
            2.0 * math.exp(0.0) - math.exp(0.0)
            + 2.0 * math.exp(-4.0) - math.exp(-2.0)
        )

        self.assertAlmostEqual(actual, expected)
        self.assertEqual(bacterium.representation, 0.0)
        self.assertEqual(other.representation, 2.0)

    def test_validation_and_copy_preserve_swarming_parameters(self):
        valid = BfoSwarmingSupportReal(1, 2, 3, 4)
        copied = valid.copy()

        self.assertIsNot(valid, copied)
        self.assertEqual(copied.attractant_depth, 1.0)
        self.assertEqual(copied.attractant_width, 2.0)
        self.assertEqual(copied.repellent_height, 3.0)
        self.assertEqual(copied.repellent_width, 4.0)

        for values in ((-1, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 0), (True, 1, 1, 1)):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    BfoSwarmingSupportReal(*values)


if __name__ == "__main__":
    unittest.main()
