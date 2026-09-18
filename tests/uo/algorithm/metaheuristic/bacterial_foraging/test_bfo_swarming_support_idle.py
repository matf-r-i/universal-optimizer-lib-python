import unittest

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_swarming_support import (
    BfoSwarmingSupportIdle,
)


class TestBfoSwarmingSupportIdle(unittest.TestCase):

    def test_interaction_value_is_zero_for_empty_population(self):
        bacterium = object()
        support = BfoSwarmingSupportIdle()

        interaction = support.interaction_value(bacterium, [])

        self.assertEqual(interaction, 0.0)

    def test_interaction_value_is_zero_for_any_population(self):
        bacterium = object()
        population = [object(), object(), object()]
        support = BfoSwarmingSupportIdle()

        interaction = support.interaction_value(bacterium, population)

        self.assertEqual(interaction, 0.0)

    def test_interaction_does_not_mutate_inputs(self):
        bacterium = {"fitness": 4.0}
        population = [{"fitness": 1.0}, {"fitness": 2.0}]
        original_bacterium = bacterium.copy()
        original_population = [item.copy() for item in population]
        support = BfoSwarmingSupportIdle()

        support.interaction_value(bacterium, population)

        self.assertEqual(bacterium, original_bacterium)
        self.assertEqual(population, original_population)

    def test_copy_returns_independent_idle_support(self):
        support = BfoSwarmingSupportIdle()

        copied = support.copy()

        self.assertIsNot(support, copied)
        self.assertIsInstance(copied, BfoSwarmingSupportIdle)
        self.assertEqual(copied.interaction_value(object(), []), 0.0)


if __name__ == "__main__":
    unittest.main()
