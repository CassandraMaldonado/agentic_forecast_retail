from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import DecisionConfig


# converting forecasts to constrained allocations to simulate outcomes.
class DecisionAgent:

    def __init__(self, config: DecisionConfig):
        self.config = config

    def propose(
        self, forecast_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        df = forecast_df.copy()
        df["base_need"] = df["forecast"]
        df["safety"] = self.config.service_level * (df["upper"] - df["forecast"])
        df["allocation"] = np.maximum(0, df["base_need"] + df["safety"])

        total_allocation = df["allocation"].sum()
        if total_allocation > self.config.capacity:
            scale = self.config.capacity / (total_allocation + 1e-8)
            df["allocation"] *= scale

        cost = df["allocation"] * self.config.unit_cost
        total_cost = cost.sum()
        if total_cost > self.config.budget:
            scale = self.config.budget / (total_cost + 1e-8)
            df["allocation"] *= scale
            cost = df["allocation"] * self.config.unit_cost

        df["cost"] = cost
        policy_info = {
            "config": asdict(self.config),
            "capacity_utilization": float(df["allocation"].sum() / (self.config.capacity + 1e-8)),
            "budget_utilization": float(cost.sum() / (self.config.budget + 1e-8)),
        }

        sim = self.simulate_outcomes(df)
        policy_info.update(sim["summary"])
        return df, policy_info

    def simulate_outcomes(self, df: pd.DataFrame, samples: int = 1000) -> Dict:
        """
        Simple simulator: sample demand from a truncated normal centered at forecast
        with spread informed by the interval width.
        """
        allocations = df["allocation"].values
        mean = df["forecast"].values
        spread = (df["upper"] - df["lower"]).values / 2 + 1e-3

        rng = np.random.default_rng(7)
        demand = rng.normal(loc=mean, scale=spread, size=(samples, len(df)))
        demand = np.clip(demand, 0, None)

        unmet = np.clip(demand - allocations, 0, None)
        over = np.clip(allocations - demand, 0, None)

        stockout_cost = unmet.sum(axis=1) * self.config.stockout_cost
        holding_cost = over.sum(axis=1) * self.config.holding_cost
        total_cost = stockout_cost + holding_cost + allocations.sum() * self.config.unit_cost

        violations = {
            "capacity_violation": float(allocations.sum() > self.config.capacity),
            "budget_violation": float((allocations * self.config.unit_cost).sum() > self.config.budget),
        }

        summary = {
            "sim_mean_cost": float(np.mean(total_cost)),
            "sim_p95_cost": float(np.quantile(total_cost, 0.95)),
            "stockout_rate": float((unmet.sum(axis=1) > 0).mean()),
            **violations,
        }

        return {"summary": summary}

