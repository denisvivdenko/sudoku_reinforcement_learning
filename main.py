import pandas as pd
import numpy as np
from dotenv import dotenv_values
import multiprocessing as mp

from src.sudoku_solver import SudokuSolver
from src.data_pipeline import parse_puzzle_clauses
from src.sudoku import Sudoku

if __name__ == "__main__":
    config = dotenv_values(".env") 
    data = pd.read_csv("src/datasets/data.csv")
    sudoku = Sudoku(parse_puzzle_clauses.fit_transform(data["puzzle"])[0])
    solver = SudokuSolver(sudoku)
    solver.solve_task()
    