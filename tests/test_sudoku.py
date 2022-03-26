import pytest
import numpy as np

from src.sudoku import Sudoku, Cell
from src.sudoku import HorizontalLineContainsValueError, VerticalLineContainsValueError, SquareContainsValueError, CellHasValueError, SquareContainsValueError

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

def test_sudoku_exception_square_contains_value() -> None:
    matrix = np.array([
        [0, 1, 2, 0, 1, 2],
        [0, 4, 5, 3, 4, 5],
        [6, 7, 8, 6, 7, 8]
    ])
    cell = Cell(row=0, column=3)
    sudoku = Sudoku(matrix)
    with pytest.raises(SquareContainsValueError):
        sudoku.insert_value(value=8, cell=cell)

def test_sudoku_successfuly_insert_value() -> None:
    matrix = np.array([
        [0, 1, 2, 0, 1, 2],
        [0, 4, 5, 3, 4, 5],
        [6, 7, 8, 6, 7, 8]
    ])
    cell = Cell(row=0, column=3)
    sudoku = Sudoku(matrix)
    sudoku.insert_value(value=9, cell=cell)

def test_cell_already_inserted() -> None:
    matrix = np.array([
        [0, 1, 2],
        [0, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    cell = Cell(row=0, column=1)
    with pytest.raises(CellHasValueError):
        sudoku.insert_value(value=6, cell=cell)

def test_empty_cells_property_passed() -> None:
    matrix = np.array([
        [0, 1, 2],
        [0, 4, 5],
        [6, 7, 0]
    ])
    sudoku = Sudoku(matrix)
    expected = [Cell(0, 0), Cell(1, 0), Cell(2, 2)]
    actual = sudoku.empty_cells
    assert expected == actual

def test_empty_cells_property_failed() -> None:
    matrix = np.array([
        [0, 1, 2],
        [0, 4, 5],
        [6, 7, 0]
    ])
    sudoku = Sudoku(matrix)
    expected = [Cell(1, 0), Cell(1, 0), Cell(2, 2)]
    actual = sudoku.empty_cells
    assert expected != actual