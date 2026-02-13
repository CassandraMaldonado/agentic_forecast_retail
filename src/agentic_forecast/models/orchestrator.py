from __future__ import annotations

import argparse
import pathlib
from typing import Dict

import pandas as pd

from .config import BacktestConfig, DataConfig, DecisionConfig, SystemConfig
from .critic import CriticAgent
from .data_quality import DataQualityAgent
from .decision import DecisionAgent
from .model_portfolio import ModelPortfolioAgent
from .signal import SignalAgent
from .uncertainty import UncertaintyAgent


def run_pipeline(
    data_path: str,
    config: SystemConfig | None = None,
    horizon: int | None = None,
) -> Dict:
    config = config or SystemConfig()
    if horizon:
        config.data.horizon = horizon

    raw = pd.read_csv(data_path)
    dq = DataQualityAgent(config.data)
    clean, dq_report = dq.validate(raw)

    signal_agent = SignalAgent(config.data)
    decomposed, signal_info = signal_agent.decompose(clean)
    features = signal_agent.build_features(decomposed)

    portfolio = ModelPortfolioAgent(config.data, config.backtest)
    backtest_results = portfolio.backtest(features)
    best = portfolio.fit_best(features)

    # last horizon rows as a placeholder.
    future = features.tail(config.data.horizon).copy()
    point_preds = portfolio.predict(future)

    best_residuals = backtest_results[best.name].residuals
    uncertainty = UncertaintyAgent(alpha=0.1)
    uncertainty.fit_residuals(best_residuals)
    lower, upper = uncertainty.intervals_from_point(point_preds)
    interval_eval = uncertainty.evaluate_intervals(
        y_true=future[config.data.target_col].values,
        lower=lower,
        upper=upper,
        nominal=0.9,
    )

    forecast_df = future[[config.data.time_col] + config.data.id_cols].copy()
    forecast_df["forecast"] = point_preds
    forecast_df["lower"] = lower
    forecast_df["upper"] = upper

    decision_agent = DecisionAgent(config.decision)
    decisions, decision_info = decision_agent.propose(forecast_df)

    critic = CriticAgent()
    critique = critic.assess(
        forecast_metrics=backtest_results[best.name].metrics,
        interval_eval=interval_eval,
        decision_info=decision_info,
    )

    return {
        "data_quality": dq_report,
        "signal": signal_info,
        "backtest": {k: v.metrics for k, v in backtest_results.items()},
        "model": best.metadata(),
        "forecast": forecast_df,
        "decisions": decisions,
        "decision_info": decision_info,
        "interval_eval": interval_eval,
        "critic": critique,
    }


def cli():
    parser = argparse.ArgumentParser(description="Agentic forecasting pipeline")
    parser.add_argument("--data", required=True, help="Path to csv data.")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon.")
    args = parser.parse_args()

    artifacts = run_pipeline(data_path=args.data, horizon=args.horizon)
    print("Backtest:", artifacts["backtest"])
    print("Model:", artifacts["model"])
    print("Interval eval:", artifacts["interval_eval"])
    print("Decision summary:", artifacts["decision_info"])
    print("Critic:", artifacts["critic"])


if __name__ == "__main__":
    cli()

