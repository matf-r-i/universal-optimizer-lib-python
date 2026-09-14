"""
..  _py_tabu_search_optimizer:

The :mod:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer` contains class
:class:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer.TabuSearchOptimizer`, that implements
:ref:`Tabu Search<Algorithm_Tabu_Search>` algorithm.
"""

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
    """
    Instance of the class :class:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer.
    TabuSearchOptimizerConstructionParameters` represents constructor parameters for Tabu Search algorithm.
    """
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
    """
    Instance of the class :class:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer.TabuSearchOptimizer`
    encapsulates :ref:`Algorithm_Tabu_Search` optimization algorithm.

    Tabu Search is a single-solution, neighborhood-based metaheuristic. At each iteration it moves to the
    best neighbor of the current solution that is not forbidden by the tabu list -- a short-term memory of
    recently applied moves -- unless that neighbor is better than the best solution found so far, in which
    case it is accepted regardless of the tabu status (aspiration criterion). Allowing temporarily
    non-improving moves lets the search escape local optima without random perturbation.
    """

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
        """
        Create new instance of class :class:`~uo.algorithm.metaheuristic.tabu_search.tabu_search_optimizer.
        TabuSearchOptimizer`. That instance implements :ref:`Tabu Search<Algorithm_Tabu_Search>` algorithm.

        :param `TabuSearchSupport` tabu_search_support: placeholder for neighborhood exploration method,
        specific for Tabu Search execution, which depends on precise solution type
        :param int tabu_tenure: number of most recent moves kept as tabu (forbidden) by the algorithm
        :param `FinishControl` finish_control: structure that control finish criteria for metaheuristic execution
        :param `Problem` problem: problem to be solved
        :param `Optional[Solution]` solution_template: initial solution of the problem
        :param `Optional[OutputControl]` output_control: structure that controls output
        :param `Optional[int]` random_seed: random seed for metaheuristic execution
        :param `Optional[AdditionalStatisticsControl]` additional_statistics_control: structure that controls
        additional statistics obtained during metaheuristic execution
        """

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
        """
        Copy the `TabuSearchOptimizer`

        :return: new `TabuSearchOptimizer` instance with the same properties
        :rtype: `TabuSearchOptimizer`
        """
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
        """
        Additional constructor, that creates new instance of class :class:`~uo.algorithm.metaheuristic.
        tabu_search.tabu_search_optimizer.TabuSearchOptimizer`.

        :param `TabuSearchOptimizerConstructionParameters` construction_tuple: tuple with all constructor
        parameters
        """
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
        """
        Property getter for the neighborhood exploration support used by Tabu Search

        :return: neighborhood exploration support used by Tabu Search
        :rtype: `TabuSearchSupport`
        """
        return self.__tabu_search_support

    @property
    def tabu_tenure(self) -> int:
        """
        Property getter for the tabu tenure parameter of the Tabu Search algorithm

        :return: tabu tenure parameter
        :rtype: int
        """
        return self.__tabu_tenure

    @property
    def tabu_list(self) -> Optional[TabuList]:
        """
        Property getter for the tabu list used during Tabu Search execution

        :return: tabu list used during execution, or `None` before initialization
        :rtype: `Optional[TabuList]`
        """
        return self.__tabu_list

    def init(self) -> None:
        """
        Initialization of the Tabu Search algorithm
        """
        self.__tabu_list = TabuList(self.tabu_tenure)
        self.current_solution = self.solution_template.copy()
        self.current_solution.copy_from(self.solution_template)
        self.current_solution.init_random(self.problem)
        self.evaluation = 1
        self.current_solution.evaluate(self.problem)
        self.best_solution = self.current_solution

    def main_loop_iteration(self) -> None:
        """
        One iteration within main loop of the Tabu Search algorithm
        """
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
        """
        String representation of the `TabuSearchOptimizer` instance

        :param delimiter: delimiter between fields
        :type delimiter: str
        :param indentation: level of indentation
        :type indentation: int, optional, default value 0
        :param indentation_symbol: indentation symbol
        :type indentation_symbol: str, optional, default value ''
        :param group_start: group start string
        :type group_start: str, optional, default value '{'
        :param group_end: group end string
        :type group_end: str, optional, default value '}'
        :return: string representation of instance that controls output
        :rtype: str
        """
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
        """
        String representation of the `TabuSearchOptimizer` instance

        :return: string representation of the `TabuSearchOptimizer` instance
        :rtype: str
        """
        s = self.string_rep('|')
        return s

    def __repr__(self) -> str:
        """
        String representation of the `TabuSearchOptimizer` instance

        :return: string representation of the `TabuSearchOptimizer` instance
        :rtype: str
        """
        s = self.string_rep('\n')
        return s

    def __format__(self, spec: str) -> str:
        """
        Formatted the TabuSearchOptimizer instance

        :param spec: str -- format specification
        :return: formatted `TabuSearchOptimizer` instance
        :rtype: str
        """
        return self.string_rep('\n', 0, '   ', '{', '}')
