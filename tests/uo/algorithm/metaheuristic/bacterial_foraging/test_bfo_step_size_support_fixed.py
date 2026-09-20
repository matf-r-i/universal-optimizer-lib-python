import unittest

from uo.algorithm.metaheuristic.bacterial_foraging.bfo_step_size_support import (
    BfoStepSizeSupportFixed,
)


class TestBfoStepSizeSupportFixed(unittest.TestCase):

    def test_constructor_stores_step_size_as_float(self):
        support = BfoStepSizeSupportFixed(2)

        self.assertEqual(support.step_size, 2.0)
        self.assertIsInstance(support.step_size, float)

    def test_initial_step_size_returns_configured_value(self):
        support = BfoStepSizeSupportFixed(0.25)

        self.assertEqual(support.initial_step_size(), 0.25)

    def test_adapt_keeps_step_size_unchanged_after_improvement_or_failure(self):
        support = BfoStepSizeSupportFixed(0.25)

        self.assertEqual(support.adapt(0.25, improved=True), 0.25)
        self.assertEqual(support.adapt(0.25, improved=False), 0.25)

    def test_copy_returns_independent_fixed_support(self):
        support = BfoStepSizeSupportFixed(0.75)

        copied = support.copy()

        self.assertIsNot(support, copied)
        self.assertIsInstance(copied, BfoStepSizeSupportFixed)
        self.assertEqual(copied.step_size, support.step_size)

    def test_constructor_rejects_non_numeric_step_size(self):
        for value in (True, "0.25", None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    BfoStepSizeSupportFixed(value)

    def test_constructor_rejects_non_positive_step_size(self):
        for value in (0, -0.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BfoStepSizeSupportFixed(value)


if __name__ == "__main__":
    unittest.main()
