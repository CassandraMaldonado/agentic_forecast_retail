from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def evaluate_forecast(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def regret_vs_oracle(costs: pd.Series, oracle_cost: float) -> float:
    return float(costs.mean() - oracle_cost)


def violation_rate(violations: pd.Series) -> float:
    return float((violations > 0).mean())

