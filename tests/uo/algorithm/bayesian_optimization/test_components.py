import unittest

import numpy as np

from uo.algorithm.bayesian_optimization.acquisition import expected_improvement
from uo.algorithm.bayesian_optimization.acquisition_optimization import maximize_acquisition
from uo.algorithm.bayesian_optimization.gaussian_process import GaussianProcessRegressor
from uo.algorithm.bayesian_optimization.kernels import RBFKernel
from uo.algorithm.bayesian_optimization.space import (
    from_unit_cube,
    sample_uniform,
    to_unit_cube,
    validate_bounds,
)


class TestBayesianComponents(unittest.TestCase):
    def test_expected_improvement_prefers_lower_mean(self):
        lower = expected_improvement(0.1, 0.2, 0.2, xi=0.0)
        higher = expected_improvement(0.5, 0.2, 0.2, xi=0.0)
        self.assertGreater(lower, higher)

    def test_expected_improvement_is_zero_without_variance(self):
        self.assertEqual(expected_improvement(0.1, 0.0, 0.2), 0.0)

    def test_rbf_kernel_matrix_is_symmetric(self):
        kernel = RBFKernel(length_scale=0.5, amplitude=1.5)
        points = np.array([[0.0, 0.0], [0.2, 0.5], [0.8, 0.1]])
        gram = kernel.matrix(points, points)
        self.assertTrue(np.allclose(gram, gram.T))
        self.assertTrue(np.allclose(kernel.diagonal(points), 1.5))

    def test_gaussian_process_interpolates_training_points(self):
        x_train = np.array([[0.0], [0.3], [0.8]], dtype=np.float64)
        y_train = np.array([0.0, 0.8, -0.1], dtype=np.float64)
        process = GaussianProcessRegressor(
            RBFKernel(0.3, 1.0),
            jitter=1e-10,
        )
        process.fit(x_train, y_train)
        mean, variance = process.predict(x_train)
        self.assertTrue(np.allclose(mean, y_train, atol=1e-4))
        self.assertTrue(np.all(variance >= 0.0))

    def test_gaussian_process_requires_fit(self):
        process = GaussianProcessRegressor(RBFKernel(0.5, 1.0))
        with self.assertRaises(RuntimeError):
            process.predict(np.array([[0.0]]))

    def test_acquisition_optimization_finds_quadratic_peak(self):
        point, value = maximize_acquisition(
            lambda x: -(float(x[0]) - 0.3) ** 2,
            np.array([[0.0, 1.0]]),
            np.random.default_rng(0),
        )
        self.assertLess(abs(float(point[0]) - 0.3), 0.1)
        self.assertGreaterEqual(value, -(0.9 - 0.3) ** 2)

    def test_space_round_trip_and_sampling(self):
        bounds = np.array([[-2.0, 2.0], [10.0, 20.0]])
        points = sample_uniform(np.random.default_rng(3), bounds, 4)
        restored = from_unit_cube(to_unit_cube(points, bounds), bounds)
        self.assertTrue(np.allclose(points, restored))
        self.assertTrue(np.all(points >= bounds[:, 0]))
        self.assertTrue(np.all(points <= bounds[:, 1]))

    def test_invalid_component_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_bounds([])
        with self.assertRaises(ValueError):
            RBFKernel(0.0, 1.0)
        with self.assertRaises(ValueError):
            maximize_acquisition(lambda x: 0.0, np.array([[0.0, 1.0]]), np.random.default_rng(), 0)


if __name__ == "__main__":
    unittest.main()
