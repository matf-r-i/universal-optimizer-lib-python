"""
..  _py_ga_selection_tournament:

The :mod:`~uo.algorithm.metaheuristic.genetic_algorithm.ga_selection_tournament` contains class
:class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaSelectionTournament`, that represents
tournament selection within the genetic algorithm.
"""

from random import sample
from typing import Optional

from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.genetic_algorithm.ga_optimizer import GaOptimizer
from uo.algorithm.metaheuristic.genetic_algorithm.ga_selection import GaSelection


class GaSelectionTournament(GaSelection):
    """
    Tournament selection.

    Every individual that is not protected by elitism is replaced by the winner of a tournament.
    `tournament_size` individuals are drawn from the population without replacement, and the one
    with the highest fitness among them is selected.

    """

    def __init__(self, tournament_size: int = 3) -> None:
        """
        Create new `GaSelectionTournament` instance

        :param tournament_size: number of individuals that take part in a single tournament
        :type tournament_size: int
        """
        if not isinstance(tournament_size, int):
            raise TypeError("Parameter 'tournament_size' must be 'int'.")
        if tournament_size < 1:
            raise ValueError("Parameter 'tournament_size' must be greater than zero.")
        self.__tournament_size: int = tournament_size

    def copy(self) -> 'GaSelectionTournament':
        """
        Copy the current object

        :return: new instance with the same properties
        :rtype: :class:`GaSelectionTournament`
        """
        return GaSelectionTournament(self.tournament_size)

    @property
    def tournament_size(self) -> int:
        """
        Getter for the size of a single tournament

        :return: number of individuals that take part in a single tournament
        :rtype: int
        """
        return self.__tournament_size

    @staticmethod
    def fitness_of(solution: Solution) -> float:
        """
        Fitness of an individual, with an individual that has none treated as the worst possible.

        :param solution: individual whose fitness is obtained
        :type solution: `Solution`
        :return: fitness of the individual
        :rtype: float
        """
        if solution.fitness_value is None:
            return float("-inf")
        return solution.fitness_value

    def selection(self, optimizer: GaOptimizer) -> None:
        """
        GA tournament selection, performed in place over the population of the optimizer.

        :param optimizer: optimizer that is executed
        :type optimizer: `GaOptimizer`
        :return:
        :rtype: None
        """
        population: Optional[list[Solution]] = optimizer.current_population
        if population is None:
            raise AttributeError("Population should exist!")
        population_size: int = len(population)
        if population_size <= 0:
            raise AttributeError("Population should contain at least one individual")
        elite_count: Optional[int] = optimizer.elite_count
        lower_limit: int = elite_count if isinstance(elite_count, int) else 0



        tournament_size: int = min(self.tournament_size, population_size)
        selected: list[Solution] = []
        for _ in range(lower_limit, population_size):
            competitors: list[Solution] = sample(population, tournament_size)
            selected.append(max(competitors, key=self.fitness_of))
        for i in range(lower_limit, population_size):
            population[i] = selected[i - lower_limit]

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
                   group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the ga selection structure

        :return: string representation of ga selection instance
        :rtype: str
        """
        return f'GaSelectionTournament{group_start}' \
               f'tournament_size={self.tournament_size}{group_end}'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
