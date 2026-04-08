"""
..  _py_moead_metrics:

The :mod:`~uo.algorithm.metaheuristic.moead.moead_metrics`
module contains quality indicators for multi-objective optimization
results, including IGD and 2D hypervolume.
"""

from __future__ import annotations

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))

import numpy as np


def igd(A: np.ndarray, Z: np.ndarray) -> float:
    """
    Compute Inverted Generational Distance.

    :param np.ndarray A: approximation set in objective space of shape ``(N, M)``
    :param np.ndarray Z: reference set in objective space of shape ``(K, M)``
    :return: IGD value
    :rtype: float
    """
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if A.ndim != 2 or Z.ndim != 2:
        raise ValueError("Parameters 'A' and 'Z' must be 2D.")
    if A.shape[1] != Z.shape[1]:
        raise ValueError("Objective dimension mismatch between 'A' and 'Z'.")

    Z_norm = np.sum(Z * Z, axis=1, keepdims=True)
    A_norm = np.sum(A * A, axis=1, keepdims=True).T
    d2 = Z_norm + A_norm - 2.0 * (Z @ A.T)
    d2 = np.maximum(d2, 0.0)

    min_d = np.sqrt(np.min(d2, axis=1))
    return float(np.mean(min_d))


def filter_nondominated_points(A: np.ndarray) -> np.ndarray:
    """
    Return nondominated subset of points for minimization problems.

    :param np.ndarray A: objective matrix of shape ``(N, M)``
    :return: nondominated subset
    :rtype: np.ndarray
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("Parameter 'A' must be 2D.")

    N = A.shape[0]
    keep = np.ones(N, dtype=bool)

    for i in range(N):
        if not keep[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if np.all(A[j] <= A[i]) and np.any(A[j] < A[i]):
                keep[i] = False
                break

    return A[keep]


def hypervolume_2d(A: np.ndarray, ref_point: tuple[float, float]) -> float:
    """
    Compute 2D hypervolume for minimization problems.

    :param np.ndarray A: objective matrix of shape ``(N, 2)``
    :param tuple[float, float] ref_point: reference point
    :return: hypervolume value
    :rtype: float
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("Parameter 'A' must be of shape (N, 2).")

    A = filter_nondominated_points(A)
    A = A[np.argsort(A[:, 0])]

    hv = 0.0
    prev_f2 = ref_point[1]

    for f1, f2 in A:
        width = ref_point[0] - f1
        height = prev_f2 - f2

        if width > 0.0 and height > 0.0:
            hv += width * height

        prev_f2 = f2

    return float(hv)