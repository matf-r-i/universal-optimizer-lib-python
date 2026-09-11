"""
The :mod:`~uo.algorithm.metaheuristic.tabu_search.tabu_list` module describes the class :class:`~uo.algorithm.metaheuristic.tabu_search.tabu_list.TabuList`.
"""

from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)

from collections import deque
from typing import Hashable


class TabuList:
    """
    This class represents short-term memory structure used by Tabu Search metaheuristic.
    It keeps track of the most recently applied moves (or solution codes), forbidding them
    from being repeated until they age out of the list.
    """

    def __init__(self, tenure: int) -> None:
        """
        Create new `TabuList` instance

        :param int tenure: maximum number of most recent moves kept as tabu (forbidden)
        """
        if not isinstance(tenure, int):
            raise TypeError('Parameter \'tenure\' must be \'int\'.')
        if tenure <= 0:
            raise ValueError('Parameter \'tenure\' must be positive.')
        self.__tenure: int = tenure
        self.__moves: deque = deque(maxlen=tenure)

    def copy(self) -> 'TabuList':
        """
        Copy the current `TabuList`

        :return: new `TabuList` instance with the same properties
        :rtype: `TabuList`
        """
        obj = TabuList(self.__tenure)
        obj.__moves = deque(self.__moves, maxlen=self.__tenure)
        return obj

    @property
    def tenure(self) -> int:
        """
        Property getter for the tenure (maximum size) of the tabu list

        :return: tenure of the tabu list
        :rtype: int
        """
        return self.__tenure

    def contains(self, move: Hashable) -> bool:
        """
        Check if the supplied move is currently forbidden (tabu)

        :param Hashable move: move to be checked
        :return: `True` if move is tabu, `False` otherwise
        :rtype: bool
        """
        return move in self.__moves

    def add(self, move: Hashable) -> None:
        """
        Add the supplied move to the tabu list. If tabu list is full, the oldest move is
        discarded to make room for the new one.

        :param Hashable move: move to be added to the tabu list
        """
        self.__moves.append(move)

    def __len__(self) -> int:
        return len(self.__moves)

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
        """
        String representation of the `TabuList` instance

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
        :return: string representation of the `TabuList` instance
        :rtype: str
        """
        s = delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += group_start
        s += 'tenure=' + str(self.__tenure) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += 'moves=' + str(list(self.__moves)) + delimiter
        for _ in range(0, indentation):
            s += indentation_symbol
        s += group_end
        return s

    def __str__(self) -> str:
        return self.string_rep('|')

    def __repr__(self) -> str:
        return self.string_rep('\n')

    def __format__(self, spec: str) -> str:
        return self.string_rep('|')
