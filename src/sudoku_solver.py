from enum import Enum
import multiprocessing
import os
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.logger import Logger
from src.sudoku import Cell, RuleViolationError, Sudoku


class SolvingStage(Enum):
    DATA_COLLECTION = 1
    TRAINING_MODELS = 2

class SudokuSolver:
    def __init__(self, sudoku: Sudoku) -> None:
        self._sudoku = sudoku
        self._epochs = multiprocessing.Value('i', 0)
        self._solving_stage = None

    def solve_task(self) -> Sudoku:
        self._solving_stage = SolvingStage.DATA_COLLECTION
        generated_plans = self._stochastic_plan_generator(models=[], processing_time=10, max_plan_len=1000, n_cores=os.cpu_count())
        Logger().debug(f"Generated plans number: {len(generated_plans)}")
        evaluated_plans = set([tuple(self._evaluate_plan(plan, self._sudoku.reset_sudoku())) for plan, _ in tqdm(generated_plans.items())])
        Logger().debug(f"Evaluated plans: {len(evaluated_plans)}")
        kpi = self._calculate_kpi(evaluated_plans)
        Logger().debug(f"KPI: {kpi}")

    def _calculate_kpi(self, plans: Set[int]) -> Dict[int, int]:
        kpi = dict()
        for plan in plans:
            if not plan:
                continue
            kpi[plan[0]] = kpi.get(plan[0], 0) + (1 + len(plan)) * len(plan) / 2
        return kpi

    def _stochastic_plan_generator(self, models: List[RandomForestClassifier], processing_time: int, max_plan_len: int, n_cores: int = 1) -> Dict[Tuple[int], int]:
        def start_generating_process(time: int, process_id: int, return_dict: dict) -> List[int]:
            start_time = time.time()
            while time.time() - start_time < processing_time:
                plan = tuple(self._generate_plan(models=models, significance_level=None, max_plan_len=max_plan_len, sudoku=self._sudoku))
                return_dict[plan] = return_dict.get(plan, 0) + 1

        manager = multiprocessing.Manager()
        return_dict = manager.dict()  # Key is plan (tuple) and value is counter of generated plans.
        processes = list()
        for core in range(n_cores):
            process = multiprocessing.Process(target=start_generating_process, args=(time, core, return_dict))
            processes.append(process)
            process.start()
        [process.join() for process in processes]
        return return_dict
    
    def _generate_plan(self, models: List[RandomForestClassifier], significance_level: List[float], max_plan_len: int, sudoku: Sudoku) -> List[int]:
        """Generates plan list which contains numbers in a certain order."""
        plan = list()
        for step, _ in enumerate(sudoku.empty_cells):
            possible_values = np.arange(1, 10)
            if step > max_plan_len:
                return plan
            try:
                right_step_probabilities = models[step].predict_proba(possible_values.reshape(9, 1))
                if any(right_step_probabilities >= significance_level[step]):
                    predicted_steps = possible_values[right_step_probabilities.flatten() >= significance_level[step]]
                    plan.append(np.random.choice(predicted_steps))
                else:
                    plan.append(np.random.choice(possible_values))
            except IndexError:
                plan.append(np.random.choice(possible_values))
        return plan

    def _evaluate_plan(self, plan: List[int], sudoku: Sudoku) -> List[int]:
        """
        Evaluates plan until error. Evaluates values in a given order.
        Parameters:
            plan (List[int]): for example, [1, 2, 3, 4]

        Returns:
            Evaluated values before error occured. 
        """
        evaluated_plan = []
        for value, empty_cell in zip(plan, sudoku.empty_cells[:len(plan) + 1]):
            try:
                sudoku.insert_value(value=value, cell=empty_cell)
                evaluated_plan.append(value)
            except RuleViolationError:
                break
        return evaluated_plan


