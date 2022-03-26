from collections import namedtuple
from distutils.log import Log
from enum import Enum
import os
from tkinter import HORIZONTAL

import numpy as np

from src.logger import Logger


class RuleViolationError(Exception):
    """Base class for other rule violation exceptions."""
    pass

class HorizontalLineContainsValueError(RuleViolationError):
    """Raised when input value that already exists in horizontal line."""
    pass

class VerticalLineContainsValueError(RuleViolationError):
    """Raised when input value that already exists in vertical line."""
    pass

class SquareContainsValueError(RuleViolationError):
    """Raised when input value that already exists in square."""
    pass

class CellHasValueError(RuleViolationError):
    """Raised when insert value into cell that already contains value."""
    pass

Cell = namedtuple("Cell", ["row", "column"])

class Axis(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

class Sudoku:
    def __init__(self, sudoku_matrix: np.array) -> None:
        """
        Parameters:
            sudoku_matrix (np.array): matrix (n x n) cells with (m x m) squares.
                cell can contain numbers [1;9] and 0 if it is empty.  
        """
        self._sudoku_matrix: np.array = sudoku_matrix

    def insert_value(self, value: int, cell: Cell) -> None:
        """
        Inserts given value to cell. 
        If it is impossible raises one of RuleViolationError exceptions.
        
        Parameters:
            cell (Cell): namedtuple Cell(row=1, column=0).
            value (int): value to input.
        """
        if self._sudoku_matrix[cell.row, cell.column] != 0:
            raise CellHasValueError()
        elif self._line_contains_value(value, cell, Axis.HORIZONTAL):
            raise HorizontalLineContainsValueError()
        elif self._line_contains_value(value, cell, Axis.VERTICAL):
            raise VerticalLineContainsValueError()
        elif self._square_contains_value(value, self._convert_cell_to_square_cell(cell)):
            raise SquareContainsValueError()
        self.sudoku_matrix[cell.row, cell.column] = value
    
    def is_solved(self) -> bool:
        """If sudoku pazzle is solved returns True."""
        if np.isin(0, self._sudoku_matrix):
            return False
        return True

    def _line_contains_value(self, value: int, cell: Cell, axis: Axis) -> bool:
        """
        If line contains given value then returns True.
        
        Parameters: 
            cell (Cell): row and column starting from 0 to (n - 1).
            value (int): searching value.
            axis (Axis): HORIZONTAL or VERTICAL
        """
        if axis == Axis.HORIZONTAL:
            if np.isin(value, self._sudoku_matrix[cell.row]):
                return True
        elif axis == Axis.VERTICAL:
            if np.isin(value, self._sudoku_matrix[:, cell.column]):
                return True
        return False

    def _convert_cell_to_square_cell(self, cell) -> int:
        """
            +-----------+
            | 1 | 2 | 3 |
            +---+---+---+
            |4  |5  | 6 |
            +---+---+---+
            | 7 | 8 | 9 |
            +---+---+---+
        """
        pass

    def _square_contains_value(self, value: int, square_cell: Cell) -> bool:
        """
        If square contains given value the returns True.
        """
        pass

