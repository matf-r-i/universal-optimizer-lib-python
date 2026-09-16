import random
import unittest

from uo.algorithm.metaheuristic.genetic_algorithm.ga_mutation_support_swap_permutation \
    import GaMutationSupportSwapPermutation
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.solution_void_representation_object import SolutionVoidObject


class OptimizerStub:
    """Minimal stand-in for a population based metaheuristic, used for evaluation accounting."""

    def __init__(self) -> None:
        self.evaluation: int = 0
        self.written: list[str] = []

    def write_output_values_if_needed(self, position: str, step: str) -> None:
        self.written.append(position)


class TestGaMutationSupportSwapPermutationCreation(unittest.TestCase):

    def test_created_with_valid_probability(self):
        support = GaMutationSupportSwapPermutation(0.25)
        self.assertEqual(support.mutation_probability, 0.25)

    def test_probability_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            GaMutationSupportSwapPermutation('a')

    def test_probability_below_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaMutationSupportSwapPermutation(-0.1)

    def test_probability_above_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaMutationSupportSwapPermutation(1.5)

    def test_copy_returns_equal_but_distinct_instance(self):
        support = GaMutationSupportSwapPermutation(0.3)
        duplicate = support.copy()
        self.assertIsNot(duplicate, support)
        self.assertEqual(duplicate.mutation_probability, support.mutation_probability)


class TestGaMutationSupportSwapPermutationMutation(unittest.TestCase):

    def setUp(self):
        self.problem = ProblemVoidMinSO("void", True, False)
        self.representation = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def make_solution(self, representation: list[int]) -> SolutionVoidObject:
        solution = SolutionVoidObject()
        solution.init_from(list(representation), self.problem)
        return solution

    def test_multiset_of_values_is_preserved(self):
        support = GaMutationSupportSwapPermutation(0.5)
        random.seed(7)
        for _ in range(300):
            solution = self.make_solution(self.representation)
            support.mutation(self.problem, solution, OptimizerStub())
            self.assertEqual(sorted(solution.representation), sorted(self.representation))

    def test_mutation_counts_exactly_one_evaluation(self):
        support = GaMutationSupportSwapPermutation(0.5)
        solution = self.make_solution(self.representation)
        optimizer = OptimizerStub()
        support.mutation(self.problem, solution, optimizer)
        self.assertEqual(optimizer.evaluation, 1)
        self.assertEqual(optimizer.written, ['before_evaluation', 'after_evaluation'])

    def test_zero_probability_leaves_representation_unchanged(self):
        support = GaMutationSupportSwapPermutation(0.0)
        solution = self.make_solution(self.representation)
        optimizer = OptimizerStub()
        support.mutation(self.problem, solution, optimizer)
        self.assertEqual(solution.representation, self.representation)
        self.assertEqual(optimizer.evaluation, 1)

    def test_probability_one_changes_representation(self):
        support = GaMutationSupportSwapPermutation(1.0)
        random.seed(11)
        for _ in range(50):
            solution = self.make_solution(self.representation)
            support.mutation(self.problem, solution, OptimizerStub())
            self.assertNotEqual(solution.representation, self.representation)

    def test_solution_without_representation_is_left_alone(self):
        support = GaMutationSupportSwapPermutation(1.0)
        solution = SolutionVoidObject()
        optimizer = OptimizerStub()
        support.mutation(self.problem, solution, optimizer)
        self.assertEqual(optimizer.evaluation, 0)

    def test_string_representations(self):
        support = GaMutationSupportSwapPermutation(0.25)
        text = support.string_rep('|')
        self.assertIn('GaMutationSupportSwapPermutation', text)
        self.assertIn('mutation_probability=0.25', text)
        self.assertEqual(str(support), text)
        self.assertEqual(format(support), text)
        self.assertEqual(repr(support), support.string_rep('\n'))


if __name__ == '__main__':
    unittest.main()
