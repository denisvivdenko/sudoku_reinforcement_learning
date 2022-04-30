from configparser import ConfigParser
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
        self.config = ConfigParser()
        self.config.read("config.ini")

    def solve_task(self, max_time: int = 10**4) -> Sudoku:
        # [6 9 5 1 8 2 5 3 7 2 8 2 1 5 7 7 3 1 8 6 1 7 9 2 1 6 9 7 4 4 5 2 6 3 6 8 3 4 7 1 3 9 5 6]
        start_time = time.time()
        training_step_index = 0
        generating_time = 10
        max_plan_length = 1
        kpi_thresholds = np.array([])
        models: List[DecisionTreeRegressor] = []
        previous_step_combinations = 9
        while time.time() - start_time < max_time:
            Logger().debug(f"Epoh: {training_step_index + 1}")
            generated_plans = self._stochastic_plan_generator(models=models, kpi_thresholds=kpi_thresholds, processing_time=generating_time, max_plan_len=max_plan_length, n_cores=os.cpu_count())
            evaluated_plans = set([tuple(self._evaluate_plan(plan, self._sudoku.reset_sudoku())) for plan, _ in tqdm(generated_plans.items())])
            current_step_kpi = self._calculate_step_kpi(training_step_index, evaluated_plans)
            
            while not self._is_kpi_detected(current_step_kpi, kpi_threshold=0):
                Logger().debug(f"Kpi is not detected. Collecting more data.")
                generated_plans = self._stochastic_plan_generator(models=models, kpi_thresholds=kpi_thresholds, processing_time=generating_time, max_plan_len=max_plan_length, n_cores=os.cpu_count())
                evaluated_plans = evaluated_plans.union(set([tuple(self._evaluate_plan(plan, self._sudoku.reset_sudoku())) for plan, _ in tqdm(generated_plans.items())]))
                current_step_kpi = self._calculate_step_kpi(training_step_index, evaluated_plans)
                Logger().debug(f"Current kpi: {current_step_kpi}")
                Logger().debug(f"Kpi threshold: {kpi_thresholds}")
                Logger().debug(f"Plans: {[plan for plan in evaluated_plans if len(plan) > training_step_index]}")

            kpi_thresholds = np.append(kpi_thresholds, self._calculate_kpi_threshold(current_step_kpi, bias_coefficient=0.6))
            # kpi_thresholds = np.append(kpi_thresholds, 0)
            models.append(self._train_model(current_step_kpi))

            Logger().debug(f"Generating time: {generating_time}")
            Logger().debug(f"Generated plans number: {len(generated_plans)}")
            Logger().debug(f"Evaluated plans: {len(evaluated_plans)}")
            Logger().debug(f"KPI: {current_step_kpi}")
            Logger().debug(f"Kpi threshold: {kpi_thresholds}")
            Logger().debug(f"Max plan lenght: {max([len(plan) for plan in evaluated_plans])}")
            Logger().debug(f"Plans: {[plan for plan in evaluated_plans if len(plan) > training_step_index]}")
            training_step_index += 1
            max_plan_length += 1
            generating_time *= np.sqrt(self.combinations / previous_step_combinations)
            previous_step_combinations = self.combinations

    def _train_model(self, kpi: Dict[int, int]) -> DecisionTreeRegressor:
        X_train, y_train = pd.DataFrame(kpi.keys()), pd.Series(kpi.values())
        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        Logger().debug(f"Score: {model.score(X_train, y_train)}")
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
        return True if sum(kpi.values()) > kpi_threshold else False

    def _calculate_kpi_threshold(self, kpi: Dict[int, int], bias_coefficient: float = 1.2) -> None:
        non_zero_kpi = [value for value in kpi.values() if value > 0]
        if len(non_zero_kpi) > 0:
            return np.median(non_zero_kpi) * bias_coefficient
        raise Exception("No kpi found.")

    def _stochastic_plan_generator(self, models: List[RandomForestClassifier], kpi_thresholds: List[float], processing_time: int, max_plan_len: int, n_cores: int = 1) -> None:
        def start_generating_process(possible_values_per_step: Dict[int, List[int]], processing_time: int, process_id: int) -> None:
            start_time = time.time()
            with open(self.get_plan_log_path(process_id=process_id), mode="w") as plan_log:
                plans = np.array([])
                while time.time() - start_time < processing_time:
                    plan = tuple(self._generate_plan(possible_values_per_step=possible_values_per_step, max_plan_len=max_plan_len, sudoku=self._sudoku))
                    plans = np.append(plans, plan)
                np.save(plan_log, plans)
                
        def get_possible_values_per_step(models: List[RandomForestClassifier], kpi_thresholds: List[float]) -> Dict[int, List[int]]:
            possible_values_per_step = {}
            for step, model in enumerate(models):
                available_values = np.arange(1, 10)
                steps_kpi = model.predict(available_values.reshape(9, 1))
                possible_values = available_values[steps_kpi > kpi_thresholds[step]]
                possible_values_per_step[step] = possible_values
            return possible_values_per_step

        processes = list()
        predicted_possible_values_per_step = get_possible_values_per_step(models, kpi_thresholds)
        self.combinations = self._count_combinations(predicted_possible_values_per_step)
        Logger().debug(f"Predicted possible values per step: {predicted_possible_values_per_step}")
        Logger().debug(f"Combinations: {self.combinations}")
        for core in range(n_cores):
            process = multiprocessing.Process(target=start_generating_process, args=(predicted_possible_values_per_step, processing_time, core))
            processes.append(process)
            process.start()
        [process.join() for process in processes]

    def _generate_plan(self, possible_values_per_step: Dict[int, List], max_plan_len: int, sudoku: Sudoku) -> List[int]:
        """Generates plan list which contains numbers in a certain order."""
        plan = list()
        for step, _ in enumerate(sudoku.empty_cells):
            if step > max_plan_len: return plan
            step_possible_values = possible_values_per_step.get(step, list(range(1, 10)))
            plan.append(np.random.choice(step_possible_values))
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

    def _count_combinations(self, possible_values_per_step: Dict[int, List[int]]) -> int:
        return np.prod([len(step_values) for step_values in possible_values_per_step.values()]) * 9

    def get_plan_log_path(self, process_id: int) -> str:
        plan_logs_dir = self.config['Paths']['session_dir'] 
        try:
            os.makedirs(plan_logs_dir)
        except OSError:
            pass 

        return os.path.join(plan_logs_dir, f"plan_log_{process_id}.npy")