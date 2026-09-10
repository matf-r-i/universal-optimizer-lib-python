from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from typing import Hashable, Optional, TypeVar

from uo.problem.problem import Problem
from uo.solution.solution import Solution
from uo.algorithm.metaheuristic.single_solution_metaheuristic import SingleSolutionMetaheuristic
from uo.algorithm.metaheuristic.tabu_search.tabu_list import TabuList
from uo.algorithm.metaheuristic.tabu_search.tabu_search_support import TabuSearchSupport

A_co = TypeVar("A_co", covariant=True)


class TabuSearchSupportStandardPermutation(TabuSearchSupport[list[int], A_co]):

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension=dimension)

    def copy(self) -> 'TabuSearchSupportStandardPermutation':
        return TabuSearchSupportStandardPermutation(self.dimension)

    def best_neighbor_move(self, problem: Problem, solution: Solution[list[int], A_co], tabu_list: TabuList,
            optimizer: SingleSolutionMetaheuristic) -> Optional[Hashable]:
        if optimizer.should_finish():
            return None
        representation: list[int] = solution.representation
        n: int = len(representation)
        if n < 2:
            return None
        start_sol: Solution = solution.copy()
        best_sol: Optional[Solution] = None
        best_move: Optional[tuple[int, int]] = None
        for i in range(n - 1):
            for j in range(i + 1, n):
                if optimizer.should_finish():
                    solution.copy_from(start_sol)
                    return None
                move: tuple[int, int] = (i, j)
                representation[i], representation[j] = representation[j], representation[i]
                optimizer.write_output_values_if_needed("before_evaluation", "b_e")
                optimizer.evaluation += 1
                solution.evaluate(problem)
                optimizer.write_output_values_if_needed("after_evaluation", "a_e")
                is_tabu: bool = tabu_list.contains(move)
                aspiration: Optional[bool] = solution.is_better(optimizer.best_solution, problem)
                if not is_tabu or aspiration:
                    if best_sol is None or solution.is_better(best_sol, problem):
                        best_sol = solution.copy()
                        best_move = move
                representation[i], representation[j] = representation[j], representation[i]
        if best_sol is None:
            solution.copy_from(start_sol)
            return None
        solution.copy_from(best_sol)
        return best_move

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
        return 'TabuSearchSupportStandardPermutation'

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
