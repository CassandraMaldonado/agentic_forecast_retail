from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .config import DataConfig
from .utils.data import add_time_features, ensure_datetime


class SignalAgent:
    """
    Decompose trend/seasonality, detect regime shifts, and create features.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def decompose(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = ensure_datetime(df, self.config.time_col)
        y = df[self.config.target_col].values
        period = 7 if self.config.freq == "D" else None
        try:
            stl = STL(y, period=period, robust=True)
            res = stl.fit()
            df["trend"] = res.trend
            df["seasonal"] = res.seasonal
            df["remainder"] = res.resid
            method = "stl"
        except Exception:
            df["trend"] = pd.Series(y).rolling(window=period or 7, min_periods=3).mean()
            df["seasonal"] = 0.0
            df["remainder"] = y - df["trend"].fillna(method="bfill")
            method = "rolling"

        regime_flag = self._regime_shift_flag(df["trend"])
        df["regime_shift"] = regime_flag
        info = {"decompose_method": method, "regime_shifts": int(regime_flag.sum())}
        return df, info

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_time_features(df, self.config.time_col)
        for lag in (7, 14, 28):
            df[f"lag_{lag}"] = df[self.config.target_col].shift(lag)
        df["rolling_mean_7"] = (
            df[self.config.target_col].rolling(7, min_periods=3).mean()
        )
        df["rolling_mean_28"] = (
            df[self.config.target_col].rolling(28, min_periods=7).mean()
        )
        df["regime_shift_lag"] = df["regime_shift"].shift(1).fillna(0)
        df = df.dropna()
        return df

    @staticmethod
    def _regime_shift_flag(trend: pd.Series, window: int = 14, threshold: float = 2.0):
        roll = trend.diff().rolling(window=window, min_periods=5).mean()
        z = (roll - roll.mean()) / (roll.std() + 1e-8)
        return (np.abs(z) > threshold).astype(int)

