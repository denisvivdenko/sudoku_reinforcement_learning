import pytest
import numpy as np

from src.sudoku import Sudoku
from src.sudoku import Cell
from src.sudoku import HorizontalLineContainsValueError, VerticalLineContainsValueError

def test_sudoku_is_solved_false() -> None:
    matrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    expected = False
    actual = sudoku.is_solved()
    assert expected == actual

def test_sudoku_is_solved_true() -> None:
    matrix = np.array([
        [2, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    expected = True
    actual = sudoku.is_solved()
    assert expected == actual

def test_sudoku_exception_inserting_value_horizontal_line() -> None:
    matrix = np.array([
        [0, 1, 2],
        [0, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    cell = Cell(row=0, column=0)
    with pytest.raises(HorizontalLineContainsValueError):
        sudoku.insert_value(value=2, cell=cell)

def test_sudoku_exception_inserting_value_vertical_line() -> None:
    matrix = np.array([
        [0, 1, 2],
        [0, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    cell = Cell(row=0, column=0)
    with pytest.raises(VerticalLineContainsValueError):
        sudoku.insert_value(value=6, cell=cell)