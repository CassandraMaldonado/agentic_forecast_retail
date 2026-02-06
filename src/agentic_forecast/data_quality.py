from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils.data import ensure_datetime


class DataQualityAgent:
    """
    Validate schema, missingness, anomalies, and leakage risk.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = ensure_datetime(df, self.config.time_col)
        report: Dict = {"config": asdict(self.config)}

        if self.config.expected_columns:
            missing_cols = [
                c for c in self.config.expected_columns if c not in df.columns
            ]
            if missing_cols:
                report["missing_columns"] = missing_cols
                raise ValueError(f"Missing columns: {missing_cols}")

        report["rows"] = len(df)
        report["duplicates"] = int(df.duplicated().sum())
        if report["duplicates"] > 0:
            df = df.drop_duplicates()

        report["missingness"] = df.isna().mean().to_dict()
        df = df.dropna(subset=[self.config.target_col])

        # Simple anomaly flag: z-score threshold on target
        target = df[self.config.target_col]
        z = (target - target.mean()) / (target.std() + 1e-8)
        df["is_anomaly"] = (np.abs(z) > 4).astype(int)
        report["anomaly_rate"] = float(df["is_anomaly"].mean())

        return df, report

    def check_leakage(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """
        Ensure validation set is strictly after train set in time.
        """
        max_train = train[self.config.time_col].max()
        min_val = val[self.config.time_col].min()
        if min_val <= max_train:
            raise ValueError(
                f"Leakage detected: validation starts {min_val} before/at train end {max_train}"
            )

