import unittest
import numpy as np

from uo.algorithm.metaheuristic.moead.moead_weights import (
    WeightSetup,
    build_weight_setup,
    neighborhood_by_euclidean,
    simplex_lattice_weights,
    weights_2d_uniform,
)


class TestMoeadWeights(unittest.TestCase):

    def test_weights_2d_uniform_shape(self):
        W = weights_2d_uniform(5)
        self.assertEqual(W.shape, (5, 2))

    def test_weights_2d_uniform_sum_to_one(self):
        W = weights_2d_uniform(5)
        sums = np.sum(W, axis=1)
        np.testing.assert_allclose(sums, np.ones(5))

    def test_simplex_lattice_weights_shape(self):
        W = simplex_lattice_weights(3, 2)
        self.assertEqual(W.shape[1], 3)

    def test_simplex_lattice_weights_sum_to_one(self):
        W = simplex_lattice_weights(3, 2)
        sums = np.sum(W, axis=1)
        np.testing.assert_allclose(sums, np.ones(W.shape[0]))

    def test_neighborhood_by_euclidean_shape(self):
        W = weights_2d_uniform(5)
        B = neighborhood_by_euclidean(W, 3)
        self.assertEqual(B.shape, (5, 3))

    def test_build_weight_setup_two_objectives(self):
        setup = build_weight_setup(n_obj=2, population_size=6, T=3)
        self.assertIsInstance(setup, WeightSetup)
        self.assertEqual(setup.W.shape, (6, 2))
        self.assertEqual(setup.B.shape, (6, 3))

    def test_build_weight_setup_three_objectives(self):
        setup = build_weight_setup(n_obj=3, H=2, T=3)
        self.assertIsInstance(setup, WeightSetup)
        self.assertEqual(setup.W.shape[1], 3)
        self.assertEqual(setup.B.shape[1], 3)

    def test_build_weight_setup_missing_population_size_for_two_objectives(self):
        with self.assertRaises(ValueError):
            build_weight_setup(n_obj=2, T=3)

    def test_build_weight_setup_missing_H_for_three_objectives(self):
        with self.assertRaises(ValueError):
            build_weight_setup(n_obj=3, T=3)


if __name__ == "__main__":
    unittest.main()