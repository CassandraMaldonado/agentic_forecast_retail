from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    return out.sort_values(time_col)


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[time_col].dt
    out["dow"] = dt.dayofweek
    out["week"] = dt.isocalendar().week.astype(int)
    out["month"] = dt.month
    out["year"] = dt.year
    out["day"] = dt.day
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    return out


def train_test_split_time(
    df: pd.DataFrame, time_col: str, cutoff: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[time_col] <= cutoff]
    test = df[df[time_col] > cutoff]
    return train, test


def rolling_origin_splits(
    df: pd.DataFrame,
    time_col: str,
    val_size: int,
    step_size: int,
    min_train: int,
    n_splits: int,
) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    unique_times = (
        df[[time_col]].drop_duplicates().sort_values(time_col)[time_col].tolist()
    )
    start_idx = min_train
    for split_idx in range(n_splits):
        train_end_idx = start_idx + split_idx * step_size
        if train_end_idx + val_size > len(unique_times):
            break
        cutoff = unique_times[train_end_idx]
        train, val = train_test_split_time(df, time_col, cutoff)
        val = val[val[time_col] <= cutoff + pd.Timedelta(days=val_size)]
        yield train, val


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

