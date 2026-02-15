from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .evaluation import coverage
from .models.quantile import QuantileGradientBoosting, conformalize_interval


# produces calibrated prediction intervals via quantile and conformal.

class UncertaintyAgent:

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.residuals: np.ndarray | None = None
        self.quantile_model = QuantileGradientBoosting()

    def fit_residuals(self, residuals: np.ndarray):
        self.residuals = residuals

    def fit_quantile(self, X: pd.DataFrame, y: pd.Series):
        self.quantile_model.fit(X, y)

    def intervals_from_point(self, point_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.residuals is None or len(self.residuals) == 0:
            band = np.quantile(np.abs(point_preds), 1 - self.alpha)
            return point_preds - band, point_preds + band
        lower, upper = point_preds, point_preds
        return conformalize_interval(lower, upper, self.residuals, self.alpha)

    def intervals_from_quantiles(
        self, X_future: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        preds = self.quantile_model.predict(X_future)
        lower = preds[min(self.quantile_model.quantiles)]
        upper = preds[max(self.quantile_model.quantiles)]
        info = {"quantiles": self.quantile_model.quantiles}
        if self.residuals is not None:
            lower, upper = conformalize_interval(lower, upper, self.residuals, self.alpha)
            info["conformal"] = True
        else:
            info["conformal"] = False
        return lower, upper, info

    def evaluate_intervals(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        nominal: float,
    ) -> Dict:
        cov = coverage(y_true, lower, upper)
        width = float(np.mean(upper - lower))
        return {"coverage": cov, "nominal": nominal, "avg_width": width}

