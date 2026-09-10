from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from dataclasses import dataclass
from typing import Optional

from uo.utils.logger import logger

from uo.problem.problem import Problem
from uo.solution.solution import Solution

from uo.algorithm.output_control import OutputControl
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.additional_statistics_control import AdditionalStatisticsControl

from uo.algorithm.metaheuristic.single_solution_metaheuristic import SingleSolutionMetaheuristic
from uo.algorithm.metaheuristic.tabu_search.tabu_list import TabuList
from uo.algorithm.metaheuristic.tabu_search.tabu_search_support import TabuSearchSupport


@dataclass
class TabuSearchOptimizerConstructionParameters:
    def __init__(self):
        pass

    tabu_search_support: TabuSearchSupport = None
    tabu_tenure: Optional[int] = None
    finish_control: Optional[FinishControl] = None
    problem: Problem = None
    solution_template: Optional[Solution] = None
    output_control: Optional[OutputControl] = None
    random_seed: Optional[int] = None
    additional_statistics_control: Optional[AdditionalStatisticsControl] = None


class TabuSearchOptimizer(SingleSolutionMetaheuristic):
    def __init__(self,
            tabu_search_support: TabuSearchSupport,
            tabu_tenure: int,
            finish_control: FinishControl,
            problem: Problem,
            solution_template: Optional[Solution],
            output_control: Optional[OutputControl] = None,
            random_seed: Optional[int] = None,
            additional_statistics_control: Optional[AdditionalStatisticsControl] = None
        ) -> None:

        if not isinstance(tabu_search_support, TabuSearchSupport):
            raise TypeError('Parameter \'tabu_search_support\' must be \'TabuSearchSupport\'.')
        if not isinstance(tabu_tenure, int):
            raise TypeError('Parameter \'tabu_tenure\' must be \'int\'.')
        if tabu_tenure <= 0:
            raise ValueError('Parameter \'tabu_tenure\' must be positive.')
        super().__init__(name='tabu_search',
                finish_control=finish_control,
                random_seed=random_seed,
                additional_statistics_control=additional_statistics_control,
                output_control=output_control,
                problem=problem,
                solution_template=solution_template)
        self.__tabu_search_support: TabuSearchSupport = tabu_search_support
        self.__tabu_tenure: int = tabu_tenure
        self.__tabu_list: Optional[TabuList] = None

    def copy(self):
        tss: Optional[TabuSearchSupport] = None
        if self.tabu_search_support is not None:
            tss = self.tabu_search_support.copy()
        fc: Optional[FinishControl] = None
        if self.finish_control is not None:
            fc = self.finish_control.copy()
        pr: Optional[Problem] = None
        if self.problem is not None:
            pr = self.problem.copy()
        st: Optional[Solution] = None
        if self.solution_template is not None:
            st = self.solution_template.copy()
        oc: Optional[OutputControl] = None
        if self.output_control is not None:
            oc = self.output_control.copy()
        asc: Optional[AdditionalStatisticsControl] = None
        if self.additional_statistics_control is not None:
            asc = self.additional_statistics_control.copy()
        obj: TabuSearchOptimizer = TabuSearchOptimizer(tss,
                                self.tabu_tenure,
                                fc,
                                pr,
                                st,
                                oc,
                                self.random_seed,
                                asc)
        return obj

    @classmethod
    def from_construction_tuple(cls, construction_tuple: TabuSearchOptimizerConstructionParameters):
        return cls(
            construction_tuple.tabu_search_support,
            construction_tuple.tabu_tenure,
            construction_tuple.finish_control,
            construction_tuple.problem,
            construction_tuple.solution_template,
            construction_tuple.output_control,
            construction_tuple.random_seed,
            construction_tuple.additional_statistics_control
        )

    @property
    def tabu_search_support(self) -> TabuSearchSupport:
        return self.__tabu_search_support

    @property
    def tabu_tenure(self) -> int:
        return self.__tabu_tenure

    @property
    def tabu_list(self) -> Optional[TabuList]:
        return self.__tabu_list

    def init(self) -> None:
        self.__tabu_list = TabuList(self.tabu_tenure)
        self.current_solution = self.solution_template.copy()
        self.current_solution.copy_from(self.solution_template)
        self.current_solution.init_random(self.problem)
        self.evaluation = 1
        self.current_solution.evaluate(self.problem)
        self.best_solution = self.current_solution

    def main_loop_iteration(self) -> None:
        self.iteration += 1
        self.write_output_values_if_needed("before_step_in_iteration", "best_neighbor_move")
        move = self.__tabu_search_support.best_neighbor_move(self.problem, self.current_solution,
                self.__tabu_list, self)
        self.write_output_values_if_needed("after_step_in_iteration", "best_neighbor_move")
        if move is None:
            return
        self.__tabu_list.add(move)
        self.update_additional_statistics_if_required(self.current_solution)
        if self.current_solution.is_better(self.best_solution, self.problem):
            self.best_solution = self.current_solution

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '', group_start: str = '{',
            group_end: str = '}') -> str:
        s = delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += group_start
        s = super().string_rep(delimiter, indentation, indentation_symbol, '', '')
        s += delimiter
        if self.current_solution is not None:
            s += 'current_solution=' + self.current_solution.string_rep(delimiter, indentation + 1,
                    indentation_symbol, group_start, group_end) + delimiter
        else:
            s += 'current_solution=None' + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += 'tabu_tenure=' + str(self.tabu_tenure) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += '__tabu_search_support=' + self.__tabu_search_support.string_rep(delimiter,
                indentation + 1, indentation_symbol, group_start, group_end) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += group_end
        return s

    def __str__(self) -> str:
        s = self.string_rep('|')
        return s

    def __repr__(self) -> str:
        s = self.string_rep('\n')
        return s

    def __format__(self, spec: str) -> str:
        return self.string_rep('\n', 0, '   ', '{', '}')
