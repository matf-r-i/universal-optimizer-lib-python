import unittest

from uo.algorithm.metaheuristic.variable_neighborhood_search.vns_ls_support_standard_fi_permutation \
    import VnsLocalSearchSupportStandardFirstImprovementPermutation
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.quality_of_solution import QualityOfSolution
from uo.solution.solution_void_representation_object import SolutionVoidObject


class InversionCountSolution(SolutionVoidObject):
    """Solution whose objective is the number of inversions, so that the optimum is a sorted list."""

    def calculate_quality_directly(self, representation, problem) -> QualityOfSolution:
        value = sum(1
                    for a in range(len(representation))
                    for b in range(a + 1, len(representation))
                    if representation[a] > representation[b])
        return QualityOfSolution(value, None, -value, None, True)

    def copy(self) -> 'InversionCountSolution':
        duplicate = InversionCountSolution()
        duplicate.copy_from(self)
        return duplicate


class OptimizerStub:
    """Minimal stand-in for a single solution metaheuristic, used by the local search support."""

    def __init__(self, k_min: int = 1, k_max: int = 5, finished: bool = False,
                 finish_after: int = -1) -> None:
        self.k_min: int = k_min
        self.k_max: int = k_max
        self.finished: bool = finished
        self.finish_after: int = finish_after
        self.asked: int = 0
        self.evaluation: int = 0
        self.written: list[str] = []

    def should_finish(self) -> bool:
        self.asked += 1
        if self.finish_after >= 0:
            return self.asked > self.finish_after
        return self.finished

    def write_output_values_if_needed(self, position: str, step: str) -> None:
        self.written.append(position)


class TestVnsLocalSearchSupportStandardFirstImprovementPermutation(unittest.TestCase):

    def setUp(self):
        self.problem = ProblemVoidMinSO("void", True, False)
        self.representation = [2, 0, 1, 0, 2, 1]
        self.support = VnsLocalSearchSupportStandardFirstImprovementPermutation(6)

    def make_solution(self, representation: list[int]) -> InversionCountSolution:
        solution = InversionCountSolution()
        solution.init_from(list(representation), self.problem)
        solution.evaluate(self.problem)
        return solution

    def test_created_with_valid_dimension(self):
        self.assertEqual(self.support.dimension, 6)

    def test_dimension_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            VnsLocalSearchSupportStandardFirstImprovementPermutation(6.0)

    def test_copy_returns_equal_but_distinct_instance(self):
        duplicate = self.support.copy()
        self.assertIsNot(duplicate, self.support)
        self.assertEqual(duplicate.dimension, self.support.dimension)

    def test_local_search_stops_at_the_first_improvement(self):
        solution = self.make_solution(self.representation)
        optimizer = OptimizerStub()
        self.assertTrue(self.support.local_search(1, self.problem, solution, optimizer))
        self.assertEqual(optimizer.evaluation, 1)
        self.assertEqual(solution.objective_value, 5)
        self.assertEqual(sorted(solution.representation), sorted(self.representation))

    def test_local_search_fails_in_local_optimum_and_restores_solution(self):
        solution = self.make_solution([0, 0, 1, 1, 2, 2])
        optimizer = OptimizerStub()
        self.assertFalse(self.support.local_search(1, self.problem, solution, optimizer))
        self.assertEqual(solution.representation, [0, 0, 1, 1, 2, 2])
        self.assertEqual(solution.objective_value, 0)

    def test_local_search_is_refused_when_optimizer_should_finish(self):
        solution = self.make_solution(self.representation)
        optimizer = OptimizerStub(finished=True)
        self.assertFalse(self.support.local_search(1, self.problem, solution, optimizer))
        self.assertEqual(solution.representation, self.representation)
        self.assertEqual(optimizer.evaluation, 0)

    def test_local_search_is_refused_when_k_is_out_of_range(self):
        solution = self.make_solution(self.representation)
        optimizer = OptimizerStub(k_min=2, k_max=4)
        self.assertFalse(self.support.local_search(1, self.problem, solution, optimizer))
        self.assertFalse(self.support.local_search(5, self.problem, solution, optimizer))
        self.assertEqual(optimizer.evaluation, 0)

    def test_local_search_is_abandoned_when_optimizer_finishes_within_the_sweep(self):
        solution = self.make_solution(self.representation)
        optimizer = OptimizerStub(finish_after=1)
        self.assertFalse(self.support.local_search(1, self.problem, solution, optimizer))
        self.assertEqual(solution.representation, self.representation)
        self.assertEqual(optimizer.evaluation, 0)

    def test_representation_of_unexpected_length_raises_value_error(self):
        solution = self.make_solution([0, 1, 2])
        with self.assertRaises(ValueError):
            self.support.local_search(1, self.problem, solution, OptimizerStub())

    def test_string_representations(self):
        text = self.support.string_rep('|')
        self.assertEqual(text, 'VnsLocalSearchSupportStandardFirstImprovementPermutation')
        self.assertEqual(str(self.support), text)
        self.assertEqual(format(self.support), text)
        self.assertEqual(repr(self.support), self.support.string_rep('\n'))


if __name__ == '__main__':
    unittest.main()
