from enum import Enum
import multiprocessing
import time
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.logger import Logger
from src.sudoku import Cell, RuleViolationError, Sudoku


class SolvingStage(Enum):
    DATA_COLLECTION: int
    TRAINING_MODELS: int

class SudokuSolver:
    def __init__(self, sudoku: Sudoku) -> None:
        self._sudoku = sudoku
        self._epochs = multiprocessing.Value('i', 0)
        self._solving_stage = None

    def solve_task(self) -> Sudoku:
        self._solving_stage = SolvingStage.DATA_COLLECTION
        

    def _stochastic_search(self, models: List[RandomForestClassifier], time: int, n_cores: int = 1) -> None:
        """Searches for the solution."""
        def _start_searching_process(max_solving_time: int, process_id: int, return_dict: dict) -> List[int]:
            evaluated_plans = []
            start_time = time.time()
            while time.time() - start_time < max_solving_time:
                plan = self._generate_plan()
                evaluated_plan = self._evaluate_plan(plan, self._sudoku)
                evaluated_plans.append(evaluated_plan)
                self._epochs.value += 1
                Logger().debug(f"PID: {process_id} EPOCHS: {self._epochs.value}")
                if len(evaluated_plan) == len(self._sudoku.empty_cells):
                    return_dict[process_id] = evaluated_plan
                    return evaluated_plan

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = list()
        for core in range(n_cores):
            process = multiprocessing.Process(target=_start_searching_process, args=(max_solving_time, core, return_dict))
            processes.append(process)
            process.start()
    
    def _generate_plan(self, models: List[RandomForestClassifier], significance_level: List[float], sudoku: Sudoku) -> List[int]:
        """Generates plan list which contains numbers in a certain order."""
        plan = list()
        for step in range(len(sudoku.empty_cells)):
            possible_values = np.arange(1, 10)
            if models[step]:
                right_step_probabilities = models[step].predict_proba(possible_values.reshape(9, 1))
                if any(right_step_probabilities >= significance_level[step]):
                    predicted_steps = possible_values[right_step_probabilities.flatten() >= significance_level[step]]
                    plan.append(np.random.choice(predicted_steps))
                else:
                    plan.append(np.random.choice(possible_values))
            else:
                plan.append(np.random.choice(possible_values))
                    

    def _evaluate_plan(self, plan: List[int], sudoku: Sudoku) -> List[int]:
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
                sudoku.insert_value(value=value, cell=empty_cell)
                evaluated_plan.append(value)
            except RuleViolationError:
                sudoku = sudoku.reset_sudoku()
                break
        sudoku = sudoku.reset_sudoku()
        return evaluated_plan


