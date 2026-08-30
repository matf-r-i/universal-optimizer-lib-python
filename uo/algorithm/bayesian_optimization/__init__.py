"""Bayesian optimization with a Gaussian-process surrogate."""

from uo.algorithm.bayesian_optimization.acquisition import expected_improvement
from uo.algorithm.bayesian_optimization.gaussian_process import GaussianProcessRegressor
from uo.algorithm.bayesian_optimization.kernels import RBFKernel
from uo.algorithm.bayesian_optimization.optimizer import (
    AcquisitionConfig,
    BayesianOptimizer,
    BayesianOptimizerConstructionParameters,
    GaussianProcessConfig,
)
from uo.algorithm.bayesian_optimization.space import (
    clip_to_bounds,
    from_unit_cube,
    sample_uniform,
    to_unit_cube,
    validate_bounds,
)

__all__ = [
    "AcquisitionConfig",
    "BayesianOptimizer",
    "BayesianOptimizerConstructionParameters",
    "GaussianProcessConfig",
    "GaussianProcessRegressor",
    "RBFKernel",
    "clip_to_bounds",
    "expected_improvement",
    "from_unit_cube",
    "sample_uniform",
    "to_unit_cube",
    "validate_bounds",
]
