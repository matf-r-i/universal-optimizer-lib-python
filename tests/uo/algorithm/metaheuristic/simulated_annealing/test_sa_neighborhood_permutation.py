import unittest

from uo.algorithm.metaheuristic.simulated_annealing.sa_neighborhood_permutation import \
    SaNeighborhoodPermutation
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.solution_void_representation_object import SolutionVoidObject


class OptimizerStub:
    """Minimal replacement for a metaheuristic optimizer, used for evaluation accounting."""


    def __init__(self) -> None:
        self.evaluation: int = 0
        self.written: list[str] = []

    def write_output_values_if_needed(self, position: str, step: str) -> None:
        self.written.append(position)


def make_solution(representation: list[int]) -> SolutionVoidObject:
    """Create a void solution carrying the supplied permutation as its representation."""
    problem = ProblemVoidMinSO("void", True, False)
    solution = SolutionVoidObject()
    solution.init_from(representation, problem)
    return solution


class TestSaNeighborhoodPermutationCreation(unittest.TestCase):

    def test_created_with_valid_parameters(self):
        neighborhood = SaNeighborhoodPermutation(9, 2)
        self.assertEqual(neighborhood.dimension, 9)
        self.assertEqual(neighborhood.k, 2)
        self.assertEqual(SaNeighborhoodPermutation(9).k, 1)

    def test_dimension_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            SaNeighborhoodPermutation(9.0, 1)

    def test_dimension_below_two_raises_value_error(self):
        with self.assertRaises(ValueError):
            SaNeighborhoodPermutation(1, 1)

    def test_k_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            SaNeighborhoodPermutation(9, 'a')

    def test_non_positive_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            SaNeighborhoodPermutation(9, 0)

    def test_copy_returns_equal_but_distinct_instance(self):
        neighborhood = SaNeighborhoodPermutation(9, 3)
        duplicate = neighborhood.copy()
        self.assertIsNot(duplicate, neighborhood)
        self.assertEqual(duplicate.dimension, neighborhood.dimension)
        self.assertEqual(duplicate.k, neighborhood.k)


class TestSaNeighborhoodPermutationMove(unittest.TestCase):

    def setUp(self):
        self.problem = ProblemVoidMinSO("void", True, False)
        self.representation = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_neighbor_is_a_different_permutation_of_the_original(self):
        neighborhood = SaNeighborhoodPermutation(9, 1)
        solution = make_solution(list(self.representation))
        neighbor = neighborhood.generate_neighbor(solution, self.problem)
        self.assertEqual(sorted(neighbor.representation), sorted(self.representation))
        self.assertNotEqual(neighbor.representation, self.representation)

    def test_original_solution_is_left_intact(self):
        neighborhood = SaNeighborhoodPermutation(9, 1)
        solution = make_solution(list(self.representation))
        neighborhood.generate_neighbor(solution, self.problem)
        self.assertEqual(solution.representation, self.representation)

    def test_several_swaps_still_yield_a_permutation(self):
        neighborhood = SaNeighborhoodPermutation(9, 4)
        solution = make_solution(list(self.representation))
        neighbor = neighborhood.generate_neighbor(solution, self.problem)
        self.assertEqual(sorted(neighbor.representation), sorted(self.representation))

    def test_representation_of_unexpected_length_raises_value_error(self):
        neighborhood = SaNeighborhoodPermutation(36, 1)
        solution = make_solution(list(self.representation))
        with self.assertRaises(ValueError):
            neighborhood.generate_neighbor(solution, self.problem)

    def test_representation_with_equal_values_yields_no_move_and_no_evaluation(self):
        neighborhood = SaNeighborhoodPermutation(4, 1)
        solution = make_solution([7, 7, 7, 7])
        optimizer = OptimizerStub()
        neighbor = neighborhood.generate_neighbor(solution, self.problem, optimizer=optimizer)
        self.assertEqual(neighbor.representation, [7, 7, 7, 7])
        self.assertEqual(optimizer.evaluation, 0)
        self.assertEqual(optimizer.written, [])

    def test_evaluation_is_accounted_within_the_optimizer(self):
        neighborhood = SaNeighborhoodPermutation(9, 1)
        solution = make_solution(list(self.representation))
        optimizer = OptimizerStub()
        neighborhood.generate_neighbor(solution, self.problem, optimizer=optimizer)
        self.assertEqual(optimizer.evaluation, 1)
        self.assertEqual(optimizer.written, ['before_evaluation', 'after_evaluation'])


class TestSaNeighborhoodPermutationStringRepresentation(unittest.TestCase):

    def test_string_representations(self):
        neighborhood = SaNeighborhoodPermutation(9, 2)
        text = neighborhood.string_rep('|')
        self.assertIn('SaNeighborhoodPermutation', text)
        self.assertIn('dimension=9', text)
        self.assertIn('k=2', text)
        self.assertEqual(str(neighborhood), text)
        self.assertEqual(format(neighborhood), text)
        self.assertEqual(repr(neighborhood), neighborhood.string_rep('\n'))


if __name__ == '__main__':
    unittest.main()