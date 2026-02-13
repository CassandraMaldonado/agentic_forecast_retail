from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig, DataConfig
from .data_quality import DataQualityAgent
from .evaluation import evaluate_forecast
from .models.baselines import BaseForecastModel, SeasonalNaive
from .models.boosted import GradientBoostedRegressor
from .utils.data import rolling_origin_splits


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    residuals: np.ndarray
    model_name: str


class ModelPortfolioAgent:
    """
    Manages multiple forecasting models and rolling-origin evaluation.
    """

    def __init__(
        self,
        data_config: DataConfig,
        backtest_config: BacktestConfig,
        models: List[BaseForecastModel] | None = None,
    ):
        self.data_config = data_config
        self.backtest_config = backtest_config
        self.models = models or [
            SeasonalNaive(),
            GradientBoostedRegressor(),
        ]
        self.best_model: BaseForecastModel | None = None
        self.last_backtest: Dict[str, BacktestResult] = {}
        self.dq = DataQualityAgent(data_config)

    def _feature_cols(self, df: pd.DataFrame) -> List[str]:
        ignore = set(
            self.data_config.id_cols
            + [self.data_config.time_col, self.data_config.target_col]
        )
        return [c for c in df.columns if c not in ignore]

    def backtest(self, df: pd.DataFrame) -> Dict[str, BacktestResult]:
        feature_cols = self._feature_cols(df)
        results: Dict[str, List[BacktestResult]] = {m.name: [] for m in self.models}

        for train, val in rolling_origin_splits(
            df,
            time_col=self.data_config.time_col,
            val_size=self.backtest_config.val_size,
            step_size=self.backtest_config.step_size,
            min_train=self.data_config.min_train_points,
            n_splits=self.backtest_config.splits,
        ):
            self.dq.check_leakage(train, val)
            X_train, y_train = train[feature_cols], train[self.data_config.target_col]
            X_val, y_val = val[feature_cols], val[self.data_config.target_col]

            for model in self.models:
                fitted = model.fit(X_train, y_train)
                preds = fitted.predict(X_val)
                metrics = evaluate_forecast(y_val.values, preds)
                residuals = y_val.values - preds
                results[model.name].append(
                    BacktestResult(metrics=metrics, residuals=residuals, model_name=model.name)
                )

        aggregated: Dict[str, BacktestResult] = {}
        for name, res_list in results.items():
            if not res_list:
                continue
            metrics_df = pd.DataFrame([r.metrics for r in res_list])
            avg_metrics = metrics_df.mean().to_dict()
            residuals = np.concatenate([r.residuals for r in res_list])
            aggregated[name] = BacktestResult(
                metrics=avg_metrics, residuals=residuals, model_name=name
            )
        self.last_backtest = aggregated
        self.best_model = self._select_best()
        return aggregated

    def _select_best(self) -> BaseForecastModel:
        if not self.last_backtest:
            raise ValueError("Run backtest before selecting best model.")
        sorted_models = sorted(
            self.last_backtest.items(), key=lambda kv: kv[1].metrics.get("wape", np.inf)
        )
        best_name = sorted_models[0][0]
        for m in self.models:
            if m.name == best_name:
                return m
        return self.models[0]

    def fit_best(self, df: pd.DataFrame) -> BaseForecastModel:
        if self.best_model is None:
            self._select_best()
        feature_cols = self._feature_cols(df)
        X, y = df[feature_cols], df[self.data_config.target_col]
        self.best_model.fit(X, y)
        return self.best_model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not fit. Call fit_best first.")
        feature_cols = self._feature_cols(df)
        return self.best_model.predict(df[feature_cols])

