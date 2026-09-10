from datetime import datetime
import unittest
import unittest.mock as mocker

from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer import TabuSearchOptimizer
from uo.algorithm.metaheuristic.tabu_search.tabu_search_support import TabuSearchSupport
from uo.problem.problem_void_min_so import ProblemVoidMinSO
from uo.solution.solution_void_representation_int import SolutionVoidInt


class TestTabuSearchOptimizer(unittest.TestCase):

    def _make_support_stub(self):
        support_stub = mocker.MagicMock(spec=TabuSearchSupport)
        support_stub.copy = mocker.Mock(return_value=support_stub)
        support_stub.best_neighbor_move = mocker.Mock(return_value=None)
        support_stub.string_rep = mocker.Mock(return_value="")
        return support_stub

    def test_tabu_search_optimizer_initialized_with_valid_parameters(self):
        finish_control = FinishControl()
        random_seed = 123
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=random_seed)
        self.assertIsInstance(tabu_search_optimizer, TabuSearchOptimizer)

    def test_tabu_search_optimizer_initialized_with_none_solution_template(self):
        finish_control = FinishControl()
        random_seed = 123
        problem = ProblemVoidMinSO("a problem", True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=None,
                random_seed=random_seed)
        self.assertIsInstance(tabu_search_optimizer, TabuSearchOptimizer)

    def test_tabu_search_optimizer_initialized_with_none_random_seed(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 0, 0, False)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=None)
        self.assertIsInstance(tabu_search_optimizer, TabuSearchOptimizer)

    def test_tabu_search_optimizer_initialized_without_tabu_search_support(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        with self.assertRaises(TypeError):
            TabuSearchOptimizer(
                    tabu_search_support=None,
                    tabu_tenure=5,
                    finish_control=finish_control,
                    problem=problem,
                    solution_template=None,
                    random_seed=123)

    def test_tabu_search_optimizer_initialized_with_wrong_type_tabu_search_support(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        with self.assertRaises(TypeError):
            TabuSearchOptimizer(
                    tabu_search_support="not appropriate type",
                    tabu_tenure=5,
                    finish_control=finish_control,
                    problem=problem,
                    solution_template=None,
                    random_seed=123)

    def test_tabu_search_optimizer_initialized_with_non_int_tabu_tenure(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        support_stub = self._make_support_stub()
        with self.assertRaises(TypeError):
            TabuSearchOptimizer(
                    tabu_search_support=support_stub,
                    tabu_tenure="5",
                    finish_control=finish_control,
                    problem=problem,
                    solution_template=None,
                    random_seed=123)

    def test_tabu_search_optimizer_initialized_with_non_positive_tabu_tenure(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        support_stub = self._make_support_stub()
        with self.assertRaises(ValueError):
            TabuSearchOptimizer(
                    tabu_search_support=support_stub,
                    tabu_tenure=0,
                    finish_control=finish_control,
                    problem=problem,
                    solution_template=None,
                    random_seed=123)

    def test_tabu_search_optimizer_initialized_with_wrong_type_finish_control(self):
        problem = ProblemVoidMinSO("a problem", True)
        support_stub = self._make_support_stub()
        with self.assertRaises(TypeError):
            TabuSearchOptimizer(
                    tabu_search_support=support_stub,
                    tabu_tenure=5,
                    finish_control="not a FinishControl",
                    problem=problem,
                    solution_template=None,
                    random_seed=123)

    def test_tabu_search_optimizer_initialized_with_wrong_type_random_seed(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        support_stub = self._make_support_stub()
        with self.assertRaises(TypeError):
            TabuSearchOptimizer(
                    tabu_search_support=support_stub,
                    tabu_tenure=5,
                    finish_control=finish_control,
                    problem=problem,
                    solution_template=None,
                    random_seed="not an int")

    def test_tabu_search_optimizer_initialized_with_wrong_type_solution_template(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        support_stub = self._make_support_stub()
        with self.assertRaises(TypeError):
            TabuSearchOptimizer(
                    tabu_search_support=support_stub,
                    tabu_tenure=5,
                    finish_control=finish_control,
                    problem=problem,
                    solution_template="not a Solution",
                    random_seed=123)

    def test_tabu_search_optimizer_init(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=None)
        tabu_search_optimizer.execution_started = datetime.now()
        tabu_search_optimizer.init()
        self.assertEqual(tabu_search_optimizer.evaluation, 1)
        self.assertIsNotNone(tabu_search_optimizer.tabu_list)
        self.assertEqual(len(tabu_search_optimizer.tabu_list), 0)

    def test_main_loop_iteration_with_no_move_found(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        support_stub.best_neighbor_move = mocker.Mock(return_value=None)
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=123)
        tabu_search_optimizer.execution_started = datetime.now()
        tabu_search_optimizer.init()
        tabu_search_optimizer.main_loop_iteration()
        self.assertEqual(tabu_search_optimizer.iteration, 1)
        self.assertEqual(len(tabu_search_optimizer.tabu_list), 0)

    def test_main_loop_iteration_with_move_found(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        support_stub.best_neighbor_move = mocker.Mock(return_value=(0, 1))
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=123)
        tabu_search_optimizer.execution_started = datetime.now()
        tabu_search_optimizer.init()
        tabu_search_optimizer.main_loop_iteration()
        self.assertEqual(tabu_search_optimizer.iteration, 1)
        self.assertTrue(tabu_search_optimizer.tabu_list.contains((0, 1)))

    def test_string_rep(self):
        finish_control = FinishControl()
        random_seed = 123
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=random_seed)
        string_representation = tabu_search_optimizer.string_rep('|')
        self.assertIn("name=tabu_search|", string_representation)
        self.assertIn("|finish_control=", string_representation)
        self.assertIn("|random_seed=123|", string_representation)
        self.assertIn("|problem=", string_representation)
        self.assertIn("|current_solution=", string_representation)
        self.assertIn("|tabu_tenure=5|", string_representation)
        self.assertIn("|__tabu_search_support=", string_representation)

    def test_str_and_repr(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=123)
        self.assertIn("name=tabu_search|", str(tabu_search_optimizer))
        self.assertIsInstance(repr(tabu_search_optimizer), str)
        self.assertIsInstance("{:}".format(tabu_search_optimizer), str)

    def test_copy(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=5,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=123)
        copied = tabu_search_optimizer.copy()
        self.assertIsInstance(copied, TabuSearchOptimizer)
        self.assertIsNot(copied, tabu_search_optimizer)
        self.assertEqual(copied.tabu_tenure, 5)

    def test_properties(self):
        finish_control = FinishControl()
        problem = ProblemVoidMinSO("a problem", True)
        solution_template = SolutionVoidInt(43, 43, 43, True)
        support_stub = self._make_support_stub()
        tabu_search_optimizer = TabuSearchOptimizer(
                tabu_search_support=support_stub,
                tabu_tenure=7,
                finish_control=finish_control,
                problem=problem,
                solution_template=solution_template,
                random_seed=123)
        self.assertEqual(tabu_search_optimizer.tabu_tenure, 7)
        self.assertIs(tabu_search_optimizer.tabu_search_support, support_stub)
        self.assertIsNone(tabu_search_optimizer.tabu_list)


if __name__ == "__main__":
    unittest.main()
