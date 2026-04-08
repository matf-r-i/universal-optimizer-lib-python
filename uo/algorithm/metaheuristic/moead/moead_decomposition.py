"""
..  _py_moead_decomposition:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_decomposition`
module contains scalarization functions used by the
:class:`~uo.algorithm.metaheuristic.moead.moead_optimizer.MoeadOptimizer`
algorithm.

Currently, the module provides Tchebyscheff scalarization.
"""

from __future__ import annotations

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))

import numpy as np


def tchebyscheff(
        F: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        eps: float = 1e-12
) -> np.ndarray:
    """
    Compute Tchebyscheff scalarization values for objective vectors.

    :param np.ndarray F: objective matrix of shape ``(N, M)``
    :param np.ndarray w: weight vector of shape ``(M,)``
    :param np.ndarray z: ideal point of shape ``(M,)``
    :param float eps: small positive value used to avoid zero weights
    :return: scalarization values of shape ``(N,)``
    :rtype: np.ndarray
    """
    F = np.asarray(F, dtype=float)
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=float)

    if F.ndim != 2:
        raise ValueError("Parameter 'F' must be 2D.")
    if w.ndim != 1:
        raise ValueError("Parameter 'w' must be 1D.")
    if z.ndim != 1:
        raise ValueError("Parameter 'z' must be 1D.")
    if F.shape[1] != w.shape[0] or w.shape[0] != z.shape[0]:
        raise ValueError("Dimension mismatch among parameters 'F', 'w', and 'z'.")

    ww = np.maximum(w, eps)
    return np.max(ww * np.abs(F - z[None, :]), axis=1)


def tchebyscheff_one(
        f: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        eps: float = 1e-12
) -> float:
    """
    Compute Tchebyscheff scalarization value for one objective vector.

    :param np.ndarray f: objective vector of shape ``(M,)``
    :param np.ndarray w: weight vector of shape ``(M,)``
    :param np.ndarray z: ideal point of shape ``(M,)``
    :param float eps: small positive value used to avoid zero weights
    :return: scalarization value
    :rtype: float
    """
    f = np.asarray(f, dtype=float)
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=float)

    if f.ndim != 1:
        raise ValueError("Parameter 'f' must be 1D.")
    if w.ndim != 1:
        raise ValueError("Parameter 'w' must be 1D.")
    if z.ndim != 1:
        raise ValueError("Parameter 'z' must be 1D.")
    if f.shape[0] != w.shape[0] or w.shape[0] != z.shape[0]:
        raise ValueError("Dimension mismatch among parameters 'f', 'w', and 'z'.")

    ww = np.maximum(w, eps)
    return float(np.max(ww * np.abs(f - z)))