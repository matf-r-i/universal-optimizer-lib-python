import random
import unittest

from uo.algorithm.metaheuristic.genetic_algorithm.ga_crossover_support_ppx_permutation \
    import GaCrossoverSupportPpxPermutation
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.solution_void_representation_object import SolutionVoidObject


class OptimizerStub:
    """Minimal stand-in for a population based metaheuristic, used for evaluation accounting."""

    def __init__(self) -> None:
        self.evaluation: int = 0
        self.written: list[str] = []

    def write_output_values_if_needed(self, position: str, step: str) -> None:
        self.written.append(position)


class TestGaCrossoverSupportPpxPermutationCreation(unittest.TestCase):

    def test_created_with_valid_probability(self):
        support = GaCrossoverSupportPpxPermutation(0.8)
        self.assertEqual(support.crossover_probability, 0.8)

    def test_probability_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            GaCrossoverSupportPpxPermutation('a')

    def test_probability_below_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaCrossoverSupportPpxPermutation(-0.1)

    def test_probability_above_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaCrossoverSupportPpxPermutation(1.5)

    def test_copy_returns_equal_but_distinct_instance(self):
        support = GaCrossoverSupportPpxPermutation(0.8)
        duplicate = support.copy()
        self.assertIsNot(duplicate, support)
        self.assertEqual(duplicate.crossover_probability, support.crossover_probability)


class TestGaCrossoverSupportPpxPermutationOffspring(unittest.TestCase):

    def test_offspring_of_two_equal_parents_equals_that_parent(self):
        parent = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        random.seed(5)
        for _ in range(50):
            self.assertEqual(
                GaCrossoverSupportPpxPermutation.offspring(parent, parent), parent)

    def test_offspring_preserves_the_multiset_of_values(self):
        parent1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        parent2 = [2, 1, 0, 2, 1, 0, 2, 1, 0]
        random.seed(3)
        for _ in range(200):
            child = GaCrossoverSupportPpxPermutation.offspring(parent1, parent2)
            self.assertEqual(sorted(child), sorted(parent1))


class TestGaCrossoverSupportPpxPermutationCrossover(unittest.TestCase):

    def setUp(self):
        self.problem = ProblemVoidMinSO("void", True, False)
        self.parent1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.parent2 = [2, 1, 0, 2, 1, 0, 2, 1, 0]
        self.support = GaCrossoverSupportPpxPermutation(1.0)

    def make_solution(self, representation: list[int] = None) -> SolutionVoidObject:
        solution = SolutionVoidObject()
        if representation is not None:
            solution.init_from(list(representation), self.problem)
        return solution

    def test_both_children_are_permutations_of_the_parents(self):
        random.seed(3)
        for _ in range(200):
            child1, child2 = self.make_solution(), self.make_solution()
            self.support.crossover(self.problem, self.make_solution(self.parent1),
                                   self.make_solution(self.parent2), child1, child2,
                                   OptimizerStub())
            self.assertEqual(sorted(child1.representation), sorted(self.parent1))
            self.assertEqual(sorted(child2.representation), sorted(self.parent1))

    def test_crossover_counts_two_evaluations(self):
        child1, child2 = self.make_solution(), self.make_solution()
        optimizer = OptimizerStub()
        self.support.crossover(self.problem, self.make_solution(self.parent1),
                               self.make_solution(self.parent2), child1, child2, optimizer)
        self.assertEqual(optimizer.evaluation, 2)
        self.assertEqual(optimizer.written, ['before_evaluation', 'after_evaluation'])

    def test_zero_probability_copies_the_parents_without_evaluating(self):
        support = GaCrossoverSupportPpxPermutation(0.0)
        child1, child2 = self.make_solution(), self.make_solution()
        optimizer = OptimizerStub()
        support.crossover(self.problem, self.make_solution(self.parent1),
                          self.make_solution(self.parent2), child1, child2, optimizer)
        self.assertEqual(child1.representation, self.parent1)
        self.assertEqual(child2.representation, self.parent2)
        self.assertEqual(optimizer.evaluation, 0)

    def test_parents_of_different_length_raise_value_error(self):
        child1, child2 = self.make_solution(), self.make_solution()
        with self.assertRaises(ValueError):
            self.support.crossover(self.problem, self.make_solution(self.parent1),
                                   self.make_solution([0, 1]), child1, child2, OptimizerStub())

    def test_parents_with_different_multiset_raise_value_error(self):
        child1, child2 = self.make_solution(), self.make_solution()
        with self.assertRaises(ValueError):
            self.support.crossover(self.problem, self.make_solution(self.parent1),
                                   self.make_solution([9] * 9), child1, child2, OptimizerStub())

    def test_parent_without_representation_makes_children_copies(self):
        child1, child2 = self.make_solution(), self.make_solution()
        optimizer = OptimizerStub()
        self.support.crossover(self.problem, self.make_solution(),
                               self.make_solution(self.parent2), child1, child2, optimizer)
        self.assertEqual(optimizer.evaluation, 0)
        self.assertEqual(child2.representation, self.parent2)

    def test_string_representations(self):
        text = self.support.string_rep('|')
        self.assertIn('GaCrossoverSupportPpxPermutation', text)
        self.assertIn('crossover_probability=1.0', text)
        self.assertEqual(str(self.support), text)
        self.assertEqual(format(self.support), text)
        self.assertEqual(repr(self.support), self.support.string_rep('\n'))


if __name__ == '__main__':
    unittest.main()
