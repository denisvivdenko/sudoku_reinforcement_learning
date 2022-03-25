import pytest
import numpy as np

from src.sudoku import Sudoku

def test_sudoku_is_solved_false():
    matrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    expected = False
    actual = sudoku.is_solved()
    assert expected == actual

def test_sudoku_is_solved_true():
    matrix = np.array([
        [2, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    sudoku = Sudoku(matrix)
    expected = True
    actual = sudoku.is_solved()
    assert expected == actual