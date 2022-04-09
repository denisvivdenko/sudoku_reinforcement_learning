from enum import Enum
import multiprocessing
import os
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
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

    def solve_task(self, max_time: int = 1000) -> Sudoku:
        start_time = time.time()
        training_step_index = 0
        kpi_thresholds = []
        models: List[DecisionTreeRegressor] = []
        generating_time = 10
        max_plan_length = 1000
        while time.time() - start_time < max_time:
            Logger().debug(f"Epoh: {training_step_index + 1}")
            generated_plans = self._stochastic_plan_generator(models=models, kpi_thresholds=kpi_thresholds, processing_time=generating_time, max_plan_len=max_plan_length, n_cores=os.cpu_count())
            evaluated_plans = set([tuple(self._evaluate_plan(plan, self._sudoku.reset_sudoku())) for plan, _ in tqdm(generated_plans.items())])
            current_step_kpi = self._calculate_step_kpi(training_step_index, evaluated_plans)
            while not self._is_kpi_detected(current_step_kpi, kpi_threshold=50):
                Logger().debug(f"Kpi is not detected. Collecting more data.")
                generated_plans = self._stochastic_plan_generator(models=models[:-2], kpi_thresholds=kpi_thresholds, processing_time=generating_time, max_plan_len=max_plan_length, n_cores=os.cpu_count())
                evaluated_plans = evaluated_plans.union(set([tuple(self._evaluate_plan(plan, self._sudoku.reset_sudoku())) for plan, _ in tqdm(generated_plans.items())]))
                current_step_kpi = self._calculate_step_kpi(training_step_index, evaluated_plans)
                Logger().debug(f"Current kpi: {current_step_kpi}")
                Logger().debug(f"Plans: {evaluated_plans}")

            kpi_thresholds.append(self._calculate_kpi_threshold(current_step_kpi, bias_coefficient=0.8))
            models.append(self._train_model(current_step_kpi))
            training_step_index += 1
            # generating_time *= 1.2
            Logger().debug(f"Evaluated plans: {len(evaluated_plans)}")
            Logger().debug(f"Generated plans number: {len(generated_plans)}")
            Logger().debug(f"KPI: {current_step_kpi}")
            Logger().debug(f"Max plan lenght: {max([len(plan) for plan in evaluated_plans])}")
            Logger().debug(f"Plans: {evaluated_plans}")

    def _train_model(self, kpi: Dict[int, int]) -> DecisionTreeRegressor:
        X_train, y_train = list(kpi.keys()), list(kpi.values())
        model = DecisionTreeRegressor(max_depth=5)
        Logger().debug(X_train)
        Logger().debug(y_train)
        model.fit(pd.DataFrame(X_train), pd.Series(y_train))
        return model

    def _calculate_step_kpi(self, training_step_index: int, plans: Set[int]) -> Dict[int, int]:
        kpi = {step_value: 0 for step_value in range(1, 10)}
        for plan in plans:
            plan = plan[training_step_index:]
            if not plan: continue
            step_value = plan[0]
            kpi[step_value] = kpi[step_value] + (1 + len(plan)) * len(plan) / 2
        return kpi

    def _is_kpi_detected(self, kpi: Dict[int, int], kpi_threshold: int) -> bool:
        return True if sum(kpi.values()) >= kpi_threshold else False

    def _calculate_kpi_threshold(self, kpi: Dict[int, int], bias_coefficient: float = 1.2) -> None:
        non_zero_kpi = [value for value in kpi.values() if value > 0]
        if len(non_zero_kpi) > 0:
            return np.mean(non_zero_kpi) * bias_coefficient
        raise Exception("No kpi found.")

    def _stochastic_plan_generator(self, models: List[RandomForestClassifier], kpi_thresholds: List[float], processing_time: int, max_plan_len: int, n_cores: int = 1) -> Dict[Tuple[int], int]:
        def start_generating_process(time: int, process_id: int, return_dict: dict) -> List[int]:
            start_time = time.time()
            while time.time() - start_time < processing_time:
                plan = tuple(self._generate_plan(models=models, kpi_thresholds=kpi_thresholds, max_plan_len=max_plan_len, sudoku=self._sudoku))
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
    
    def _generate_plan(self, models: List[RandomForestClassifier], kpi_thresholds: List[float], max_plan_len: int, sudoku: Sudoku) -> List[int]:
        """Generates plan list which contains numbers in a certain order."""
        plan = list()
        for step, _ in enumerate(sudoku.empty_cells):
            possible_values = np.arange(1, 10)
            if step > max_plan_len:
                return plan
            try:
                step_kpi = models[step].predict(possible_values.reshape(9, 1))
                if any(step_kpi >= kpi_thresholds[step]):
                    predicted_steps = possible_values[step_kpi.flatten() >= kpi_thresholds[step]]
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


