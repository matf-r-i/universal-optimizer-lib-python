"""
..  _py_moead_weights:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_weights`
module contains functions for generating MOEA/D weight vectors and
their neighborhoods.
"""

from __future__ import annotations

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


def weights_2d_uniform(N: int) -> np.ndarray:
    """
    Generate uniformly distributed 2D weight vectors.

    :param int N: number of weight vectors
    :return: matrix of shape ``(N, 2)``
    :rtype: np.ndarray
    """
    if not isinstance(N, int):
        raise TypeError("Parameter 'N' must be 'int'.")
    if N < 2:
        raise ValueError("Parameter 'N' must be >= 2.")

    w1 = np.linspace(0.0, 1.0, N)
    return np.stack([w1, 1.0 - w1], axis=1)


def simplex_lattice_weights(M: int, H: int) -> np.ndarray:
    """
    Generate simplex-lattice design weights.

    :param int M: number of objectives
    :param int H: lattice parameter
    :return: matrix of shape ``(N, M)``
    :rtype: np.ndarray
    """
    if not isinstance(M, int):
        raise TypeError("Parameter 'M' must be 'int'.")
    if not isinstance(H, int):
        raise TypeError("Parameter 'H' must be 'int'.")
    if M < 2:
        raise ValueError("Parameter 'M' must be >= 2.")
    if H < 1:
        raise ValueError("Parameter 'H' must be >= 1.")

    N = comb(H + M - 1, M - 1)
    W = np.zeros((N, M), dtype=float)

    idx = 0
    slots = H + M - 1

    for bars in combinations(range(slots), M - 1):
        prev = -1
        ks: list[int] = []
        for b in bars:
            ks.append(b - prev - 1)
            prev = b
        ks.append(slots - prev - 1)
        W[idx, :] = np.array(ks, dtype=int) / float(H)
        idx += 1

    return W


def neighborhood_by_euclidean(W: np.ndarray, T: int) -> np.ndarray:
    """
    Build neighborhood matrix from weight vectors using Euclidean distance.

    :param np.ndarray W: weight matrix of shape ``(N, M)``
    :param int T: neighborhood size
    :return: neighborhood index matrix of shape ``(N, T)``
    :rtype: np.ndarray
    """
    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("Parameter 'W' must be 2D.")
    if not isinstance(T, int):
        raise TypeError("Parameter 'T' must be 'int'.")

    N = W.shape[0]
    if T < 1 or T > N:
        raise ValueError("Parameter 'T' must belong to interval [1, N].")

    norms = np.sum(W * W, axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (W @ W.T)
    d2 = np.maximum(d2, 0.0)

    return np.argsort(d2, axis=1)[:, :T]


@dataclass(frozen=True)
class WeightSetup:
    """
    Container that stores MOEA/D weight vectors and neighborhoods.

    :ivar W: weight-vector matrix
    :vartype W: np.ndarray
    :ivar B: neighborhood matrix
    :vartype B: np.ndarray
    """
    W: np.ndarray
    B: np.ndarray


def build_weight_setup(
        n_obj: int,
        population_size: int | None = None,
        H: int | None = None,
        T: int = 20
) -> WeightSetup:
    """
    Build weight vectors and neighborhoods for MOEA/D.

    :param int n_obj: number of objectives
    :param int population_size: number of weight vectors for ``n_obj == 2``
    :param int H: lattice parameter for ``n_obj >= 3``
    :param int T: neighborhood size
    :return: weight setup object
    :rtype: WeightSetup
    """
    if not isinstance(n_obj, int):
        raise TypeError("Parameter 'n_obj' must be 'int'.")
    if n_obj < 2:
        raise ValueError("Parameter 'n_obj' must be >= 2.")

    if n_obj == 2:
        if population_size is None:
            raise ValueError("Parameter 'population_size' must be provided for two-objective problems.")
        W = weights_2d_uniform(population_size)
    else:
        if H is None:
            raise ValueError("Parameter 'H' must be provided for problems with three or more objectives.")
        W = simplex_lattice_weights(n_obj, H)

    B = neighborhood_by_euclidean(W, min(T, W.shape[0]))
    return WeightSetup(W=W, B=B)