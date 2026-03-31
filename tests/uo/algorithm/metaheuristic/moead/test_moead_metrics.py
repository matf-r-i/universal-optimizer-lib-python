import unittest
import numpy as np

from uo.algorithm.metaheuristic.moead.moead_metrics import (
    filter_nondominated_points,
    hypervolume_2d,
    igd,
)


class TestMoeadMetrics(unittest.TestCase):

    def test_igd(self):
        A = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        Z = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])

        value = igd(A, Z)
        self.assertAlmostEqual(value, 0.0)

    def test_igd_invalid_dimensions(self):
        A = np.array([1.0, 2.0])
        Z = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            igd(A, Z)

    def test_filter_nondominated_points(self):
        A = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0]
        ])

        nd = filter_nondominated_points(A)

        self.assertEqual(nd.shape[0], 2)

    def test_hypervolume_2d(self):
        A = np.array([
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0]
        ])
        ref_point = (4.0, 4.0)

        value = hypervolume_2d(A, ref_point)

        self.assertGreater(value, 0.0)

    def test_hypervolume_2d_invalid_shape(self):
        A = np.array([[1.0, 2.0, 3.0]])
        ref_point = (4.0, 4.0)

        with self.assertRaises(ValueError):
            hypervolume_2d(A, ref_point)


if __name__ == "__main__":
    unittest.main()