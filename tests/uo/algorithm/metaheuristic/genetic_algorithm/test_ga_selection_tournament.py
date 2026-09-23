import random
import unittest

from uo.algorithm.metaheuristic.genetic_algorithm.ga_selection_tournament import \
    GaSelectionTournament
from uo.solution.solution_void_representation_object import SolutionVoidObject


class OptimizerStub:
    """Minimal stand-in for a genetic algorithm optimizer, holding a population."""

    def __init__(self, population: list, elite_count: int = 0) -> None:
        self.current_population: list = population
        self.elite_count: int = elite_count


def individual(fitness) -> SolutionVoidObject:
    """Create an individual carrying only the supplied fitness value."""
    return SolutionVoidObject(fitness_value=fitness)


def fitness_values(population: list) -> list:
    """Fitness of every individual of the population."""
    return [GaSelectionTournament.fitness_of(item) for item in population]


class TestGaSelectionTournamentCreation(unittest.TestCase):

    def test_created_with_valid_size(self):
        self.assertEqual(GaSelectionTournament(5).tournament_size, 5)

    def test_default_size_is_three(self):
        self.assertEqual(GaSelectionTournament().tournament_size, 3)

    def test_size_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            GaSelectionTournament(2.5)

    def test_non_positive_size_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaSelectionTournament(0)

    def test_copy_returns_equal_but_distinct_instance(self):
        selection = GaSelectionTournament(4)
        duplicate = selection.copy()
        self.assertIsNot(duplicate, selection)
        self.assertEqual(duplicate.tournament_size, selection.tournament_size)


class TestGaSelectionTournamentSelection(unittest.TestCase):

    def setUp(self):
        random.seed(1)

    def average_selected_fitness(self, tournament_size: int, runs: int = 200) -> float:
        total = 0.0
        for _ in range(runs):
            optimizer = OptimizerStub([individual(float(i)) for i in range(100)])
            GaSelectionTournament(tournament_size).selection(optimizer)
            total += sum(fitness_values(optimizer.current_population)) / 100
        return total / runs

    def test_selection_pressure_grows_with_tournament_size(self):
        averages = [self.average_selected_fitness(size) for size in (1, 2, 3, 5, 10)]
        for smaller, larger in zip(averages, averages[1:]):
            self.assertLess(smaller, larger)

    def test_tournament_of_size_one_keeps_the_population_average(self):
        self.assertAlmostEqual(self.average_selected_fitness(1), 49.5, delta=2.0)

    def test_tournament_of_population_size_always_selects_the_best(self):
        optimizer = OptimizerStub([individual(float(i)) for i in range(4)])
        GaSelectionTournament(4).selection(optimizer)
        self.assertEqual(fitness_values(optimizer.current_population), [3.0] * 4)

    def test_tournament_larger_than_population_is_limited(self):
        optimizer = OptimizerStub([individual(float(i)) for i in range(4)])
        GaSelectionTournament(100).selection(optimizer)
        self.assertEqual(fitness_values(optimizer.current_population), [3.0] * 4)

    def test_elite_individuals_are_left_untouched(self):
        optimizer = OptimizerStub([individual(float(i)) for i in range(10)], elite_count=3)
        GaSelectionTournament(3).selection(optimizer)
        self.assertEqual(fitness_values(optimizer.current_population)[:3], [0.0, 1.0, 2.0])

    def test_elite_count_that_is_not_integer_is_treated_as_zero(self):
        optimizer = OptimizerStub([individual(float(i)) for i in range(4)], elite_count=None)
        GaSelectionTournament(4).selection(optimizer)
        self.assertEqual(fitness_values(optimizer.current_population), [3.0] * 4)

    def test_individual_without_fitness_is_treated_as_the_worst(self):
        optimizer = OptimizerStub([individual(None), individual(5.0)])
        GaSelectionTournament(2).selection(optimizer)
        self.assertEqual(fitness_values(optimizer.current_population), [5.0, 5.0])

    def test_missing_population_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            GaSelectionTournament(3).selection(OptimizerStub(None))

    def test_empty_population_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            GaSelectionTournament(3).selection(OptimizerStub([]))

    def test_string_representations(self):
        selection = GaSelectionTournament(4)
        text = selection.string_rep('|')
        self.assertIn('GaSelectionTournament', text)
        self.assertIn('tournament_size=4', text)
        self.assertEqual(str(selection), text)
        self.assertEqual(format(selection), text)
        self.assertEqual(repr(selection), selection.string_rep('\n'))


if __name__ == '__main__':
    unittest.main()
