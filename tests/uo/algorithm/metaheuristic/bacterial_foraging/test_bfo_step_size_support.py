import unittest

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_step_size_support import (
    BfoStepSizeSupportAdaptive,
)


class TestBfoStepSizeSupportAdaptive(unittest.TestCase):

    def test_adapt_increases_decreases_and_clamps(self):
        support = BfoStepSizeSupportAdaptive(
            initial_step_size=1.0,
            minimum_step_size=0.5,
            maximum_step_size=2.0,
            increase_factor=2.0,
            decrease_factor=0.5,
        )

        self.assertEqual(support.initial_step_size(), 1.0)
        self.assertEqual(support.adapt(1.0, True), 2.0)
        self.assertEqual(support.adapt(2.0, True), 2.0)
        self.assertEqual(support.adapt(1.0, False), 0.5)
        self.assertEqual(support.adapt(0.5, False), 0.5)

    def test_validation_rejects_invalid_ranges_factors_and_flags(self):
        invalid_parameters = (
            (1.0, 0.0, 2.0, 1.05, 0.95),
            (1.0, 2.0, 1.0, 1.05, 0.95),
            (3.0, 0.5, 2.0, 1.05, 0.95),
            (1.0, 0.5, 2.0, 0.9, 0.95),
            (1.0, 0.5, 2.0, 1.05, 0.0),
        )
        for values in invalid_parameters:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    BfoStepSizeSupportAdaptive(*values)

        support = BfoStepSizeSupportAdaptive(1.0, 0.5, 2.0)
        with self.assertRaises(TypeError):
            support.adapt(1.0, 1)

    def test_copy_preserves_adaptive_configuration(self):
        support = BfoStepSizeSupportAdaptive(1.0, 0.5, 2.0, 1.2, 0.8)
        copied = support.copy()

        self.assertIsNot(support, copied)
        self.assertEqual(copied.initial_step_size_value, 1.0)
        self.assertEqual(copied.minimum_step_size, 0.5)
        self.assertEqual(copied.maximum_step_size, 2.0)
        self.assertEqual(copied.increase_factor, 1.2)
        self.assertEqual(copied.decrease_factor, 0.8)


if __name__ == "__main__":
    unittest.main()
