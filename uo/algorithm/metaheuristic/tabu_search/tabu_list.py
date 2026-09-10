from pathlib import Path
directory = Path(__file__).resolve()
import sys
sys.path.append(directory.parent)

from collections import deque
from typing import Hashable


class TabuList:

    def __init__(self, tenure: int) -> None:
        if not isinstance(tenure, int):
            raise TypeError('Parameter \'tenure\' must be \'int\'.')
        if tenure <= 0:
            raise ValueError('Parameter \'tenure\' must be positive.')
        self.__tenure: int = tenure
        self.__moves: deque = deque(maxlen=tenure)

    def copy(self) -> 'TabuList':
        obj = TabuList(self.__tenure)
        obj.__moves = deque(self.__moves, maxlen=self.__tenure)
        return obj

    @property
    def tenure(self) -> int:
        return self.__tenure

    def contains(self, move: Hashable) -> bool:
        return move in self.__moves

    def add(self, move: Hashable) -> None:
        self.__moves.append(move)

    def __len__(self) -> int:
        return len(self.__moves)

    def string_rep(self, delimiter: str, indentation: int = 0, indentation_symbol: str = '',
            group_start: str = '{', group_end: str = '}') -> str:
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
