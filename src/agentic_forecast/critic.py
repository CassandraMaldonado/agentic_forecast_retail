from __future__ import annotations

from typing import Dict, List

# Evaluates forecast and decision outcomes then it suggests next actions.
class CriticAgent:

    def __init__(self, wape_threshold: float = 0.2, coverage_tolerance: float = 0.05):
        self.wape_threshold = wape_threshold
        self.coverage_tolerance = coverage_tolerance

    def assess(
        self,
        forecast_metrics: Dict[str, float],
        interval_eval: Dict[str, float],
        decision_info: Dict,
    ) -> Dict:
        recommendations: List[str] = []

        if forecast_metrics.get("wape", 1.0) > self.wape_threshold:
            recommendations.append("Improve features or switch to boosted model; WAPE above threshold.")

        coverage_gap = abs(interval_eval.get("coverage", 0) - interval_eval.get("nominal", 0))
        if coverage_gap > self.coverage_tolerance:
            recommendations.append("Recalibrate intervals (conformal) to tighten coverage.")

        if decision_info.get("stockout_rate", 0) > (1 - decision_info.get("config", {}).get("service_level", 0.9)):
            recommendations.append("Increase service level or safety factor to reduce stockouts.")

        if decision_info.get("budget_violation", 0) > 0:
            recommendations.append("Tighten allocation or unit costs to avoid budget breach.")

        if not recommendations:
            recommendations.append("Continue current policy; metrics within targets.")

        return {"recommendations": recommendations}

