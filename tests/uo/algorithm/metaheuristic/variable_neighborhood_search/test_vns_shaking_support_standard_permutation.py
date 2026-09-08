import unittest

from uo.algorithm.metaheuristic.variable_neighborhood_search.vns_shaking_support_standard_permutation \
    import VnsShakingSupportStandardPermutation
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.solution_void_representation_object import SolutionVoidObject


class OptimizerStub:
    """Minimal replacement for a single solution metaheuristic, used by the shaking support."""

    def __init__(self, k_min: int = 1, k_max: int = 5, finished: bool = False) -> None:
        self.k_min: int = k_min
        self.k_max: int = k_max
        self.finished: bool = finished
        self.evaluation: int = 0
        self.written: list[str] = []

    def should_finish(self) -> bool:
        return self.finished

    def write_output_values_if_needed(self, position: str, step: str) -> None:
        self.written.append(position)


class TestVnsShakingSupportStandardPermutation(unittest.TestCase):

    def setUp(self):
        self.problem = ProblemVoidMinSO("void", True, False)
        self.representation = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.support = VnsShakingSupportStandardPermutation(9)

    def make_solution(self, representation: list[int]) -> SolutionVoidObject:
        solution = SolutionVoidObject()
        solution.init_from(representation, self.problem)
        return solution

    def test_created_with_valid_dimension(self):
        self.assertEqual(self.support.dimension, 9)

    def test_dimension_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            VnsShakingSupportStandardPermutation(9.0)

    def test_copy_returns_equal_but_distinct_instance(self):
        duplicate = self.support.copy()
        self.assertIsNot(duplicate, self.support)
        self.assertEqual(duplicate.dimension, self.support.dimension)

    def test_shaking_yields_a_different_permutation_and_counts_evaluation(self):
        solution = self.make_solution(list(self.representation))
        optimizer = OptimizerStub()
        self.assertTrue(self.support.shaking(3, self.problem, solution, optimizer))
        self.assertEqual(sorted(solution.representation), sorted(self.representation))
        self.assertNotEqual(solution.representation, self.representation)
        self.assertEqual(optimizer.evaluation, 1)
        self.assertEqual(optimizer.written, ['before_evaluation', 'after_evaluation'])

    def test_shaking_is_refused_when_optimizer_should_finish(self):
        solution = self.make_solution(list(self.representation))
        optimizer = OptimizerStub(finished=True)
        self.assertFalse(self.support.shaking(3, self.problem, solution, optimizer))
        self.assertEqual(solution.representation, self.representation)
        self.assertEqual(optimizer.evaluation, 0)

    def test_shaking_is_refused_when_k_is_out_of_range(self):
        solution = self.make_solution(list(self.representation))
        optimizer = OptimizerStub(k_min=2, k_max=4)
        self.assertFalse(self.support.shaking(1, self.problem, solution, optimizer))
        self.assertFalse(self.support.shaking(5, self.problem, solution, optimizer))
        self.assertEqual(optimizer.evaluation, 0)

    def test_shaking_is_refused_when_all_values_are_equal(self):
        solution = self.make_solution([7] * 9)
        optimizer = OptimizerStub()
        self.assertFalse(self.support.shaking(2, self.problem, solution, optimizer))
        self.assertEqual(solution.representation, [7] * 9)
        self.assertEqual(optimizer.evaluation, 0)

    def test_representation_of_unexpected_length_raises_value_error(self):
        solution = self.make_solution([0, 1, 2])
        with self.assertRaises(ValueError):
            self.support.shaking(1, self.problem, solution, OptimizerStub())

    def test_string_representations(self):
        text = self.support.string_rep('|')
        self.assertEqual(text, 'VnsShakingSupportStandardPermutation')
        self.assertEqual(str(self.support), text)
        self.assertEqual(format(self.support), text)
        self.assertEqual(repr(self.support), self.support.string_rep('\n'))


if __name__ == '__main__':
    unittest.main()