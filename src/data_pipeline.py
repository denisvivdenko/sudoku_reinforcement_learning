import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from src.logger import Logger

logger = Logger()

@FunctionTransformer
def parse_puzzle_clauses(clauses: pd.Series) -> np.ndarray:
    def _parse_clause(clause: str) -> np.ndarray:
        size = int(np.sqrt(len(clause)))
        numbers = np.array([int(number) for number in clause])
        return numbers.reshape((size, size))
    return np.array([_parse_clause(clause) for clause in clauses.values])

if __name__ == "__main__":
    data = pd.read_csv("src/datasets/data.csv")
    logger.debug(parse_puzzle_clauses.fit_transform(data["puzzle"]))

