import unittest
import numpy as np

from uo.algorithm.metaheuristic.moead.moead_decomposition import (
    tchebyscheff,
    tchebyscheff_one,
)


class TestMoeadDecomposition(unittest.TestCase):

    def test_tchebyscheff_one(self):
        f = np.array([2.0, 3.0])
        w = np.array([0.5, 0.5])
        z = np.array([1.0, 1.0])

        value = tchebyscheff_one(f, w, z)

        self.assertAlmostEqual(value, 1.0)

    def test_tchebyscheff_matrix(self):
        F = np.array([
            [2.0, 3.0],
            [1.5, 2.0]
        ])
        w = np.array([0.5, 0.5])
        z = np.array([1.0, 1.0])

        values = tchebyscheff(F, w, z)

        self.assertEqual(values.shape, (2,))
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], 0.5)

    def test_tchebyscheff_invalid_F_dimension(self):
        F = np.array([1.0, 2.0])
        w = np.array([0.5, 0.5])
        z = np.array([0.0, 0.0])

        with self.assertRaises(ValueError):
            tchebyscheff(F, w, z)

    def test_tchebyscheff_invalid_dimension_mismatch(self):
        F = np.array([[1.0, 2.0]])
        w = np.array([0.5, 0.3, 0.2])
        z = np.array([0.0, 0.0])

        with self.assertRaises(ValueError):
            tchebyscheff(F, w, z)


if __name__ == "__main__":
    unittest.main()