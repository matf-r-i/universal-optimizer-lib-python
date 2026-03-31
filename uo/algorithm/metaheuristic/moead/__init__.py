"""
..  _py_moead:

The :mod:`~uo.algorithm.metaheuristic.moead`
package contains implementation of the Multi-Objective Evolutionary
Algorithm based on Decomposition (MOEA/D), together with supporting
functionality for decomposition, weight-vector generation, variation
support, and quality indicators.
"""

from uo.algorithm.metaheuristic.moead.moead_decomposition import (
    tchebyscheff,
    tchebyscheff_one,
)
from uo.algorithm.metaheuristic.moead.moead_metrics import (
    filter_nondominated_points,
    hypervolume_2d,
    igd,
)
from uo.algorithm.metaheuristic.moead.moead_optimizer import MoeadOptimizer
from uo.algorithm.metaheuristic.moead.moead_variation_support import MoeadVariationSupport
from uo.algorithm.metaheuristic.moead.moead_variation_support_real import MoeadVariationSupportReal
from uo.algorithm.metaheuristic.moead.moead_variation_support_binary import MoeadVariationSupportBinary
from uo.algorithm.metaheuristic.moead.moead_weights import (
    WeightSetup,
    build_weight_setup,
    neighborhood_by_euclidean,
    simplex_lattice_weights,
    weights_2d_uniform,
)

__all__ = [
    "MoeadOptimizer",
    "MoeadVariationSupport",
    "MoeadVariationSupportReal",
    "MoeadVariationSupportBinary",
    "WeightSetup",
    "build_weight_setup",
    "weights_2d_uniform",
    "simplex_lattice_weights",
    "neighborhood_by_euclidean",
    "tchebyscheff",
    "tchebyscheff_one",
    "igd",
    "filter_nondominated_points",
    "hypervolume_2d",
]