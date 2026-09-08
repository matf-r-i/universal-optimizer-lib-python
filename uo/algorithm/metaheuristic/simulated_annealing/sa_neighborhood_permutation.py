
from uo.algorithm.metaheuristic.simulated_annealing.sa_neighborhood import SaNeighborhood

class SaNeighborhoodPermutation(SaNeighborhood):
    def __init__(self, dimension: int, k: int = 1) -> None:
        """
        Create new `SaNeighboorhoodPermutation` instance

        :param dimension: length of the permutation used as representation 
        :type dimension: int
        :param k: number of swaps that form a single move, defaults to 1
        :type k: int, optional
        """

        if not isinstance(dimension, int):
            raise TypeError("Parameter 'dimension' must be 'int'.")
        if dimension < 2:
            raise ValueError("Parameter 'dimension' must be at least 2.")
        if not isinstance(k, int):
            raise TypeError("Parameter 'k' must be 'int'.")
        if k < 1:
            raise ValueError("Parameter 'k' must be greater than zero.")
        self.dimension: int = dimension
        self.k: int = k
    