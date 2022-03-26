import pandas as pd
import numpy as np

from src.data_pipeline import parse_puzzle_clauses

def test_parse_puzzle_clauses_passes() -> None:
    puzzles = pd.Series(["0123", "4567"])
    expected = np.array([
        [[0, 1],
         [2, 3]],
        [[4, 5],
         [6, 7]]])
    actual = parse_puzzle_clauses.fit_transform(puzzles)
    assert np.array_equal(actual, expected)

def test_parse_puzzle_clauses_fails() -> None:
    puzzles = pd.Series(["0123", "4567"])
    expected = np.array([
        [[0, 1],
         [2, 3]],
        [[4, 7],
         [6, 7]]])
    actual = parse_puzzle_clauses.fit_transform(puzzles)
    assert not np.array_equal(actual, expected)