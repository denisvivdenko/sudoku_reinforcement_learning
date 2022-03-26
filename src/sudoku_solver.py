import time
from typing import List
import numpy as np

from src.sudoku import Sudoku, Cell
from src.sudoku import RuleViolationError
from src.logger import Logger

class SudokuSolver:
    def __init__(self, sudoku: Sudoku) -> None:
        self._sudoku = sudoku

    def stochastic_search(self, max_solving_time: int = 10**3) -> None:
        """Searches for the solution."""

        def _generate_plan(available_values: List[int], plan_length: int) -> List[int]:
            """Generates plan which contains numbers in a certain order."""
            return np.random.choice(available_values, size=plan_length)

        def _evaluate_plan(plan: List[int], sudoku: Sudoku) -> List[int]:
            """
            Evaluates plan until error. Evaluates values in a given order.
            Parameters:
                plan (List[int]): for example, [1, 2, 3, 4]

            Returns:
                Evaluated values before error occured. 
            """
            evaluated_plan = []
            for value, empty_cell in zip(plan, sudoku.empty_cells):
                try:
                    self._sudoku.insert_value(value=value, cell=empty_cell)
                    evaluated_plan.append(value)
                except RuleViolationError:
                    return evaluated_plan
            return evaluated_plan

        available_values = np.arange(1, 10)  # values in range [1;9]
        evaluated_plans = []
        start_time = time.time()
        while True:
            plan = _generate_plan(available_values, plan_length=len(self._sudoku.empty_cells))
            evaluated_plan = _evaluate_plan(plan, self._sudoku)
            evaluated_plans.append(evaluated_plan)
            self._sudoku = self._sudoku.reset_sudoku()
            Logger().debug(evaluated_plan)
            if time.time() - start_time > max_solving_time:
                break        