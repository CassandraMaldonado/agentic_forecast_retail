Architecture
============

System overview
---------------
The system is organized as a loop of specialized agents that transform data into decisions, evaluate outcomes, and adapt:

1) DataQualityAgent: validate schema/types, missingness, duplicates, anomalies, and leakage risks given time splits.
2) SignalAgent: decompose trend/seasonality/events, detect regime shifts, and produce features for modeling.
3) ModelPortfolioAgent: train/manage multiple forecasting models and run rolling-origin backtests.
4) UncertaintyAgent: calibrate prediction intervals (quantile and conformal) and score coverage.
5) DecisionAgent: translate forecasts + intervals into constrained decisions (capacity/budget) with risk-aware objectives.
6) CriticAgent: evaluate forecast and decision outcomes, propose next iteration (model/interval/policy adjustments).

Interaction diagram (textual)
-----------------------------
- DataQualityAgent -> (clean data, validation report) -> SignalAgent
- SignalAgent -> (feature table, regime flags) -> ModelPortfolioAgent
- ModelPortfolioAgent -> (point forecasts, validation residuals) -> UncertaintyAgent
- UncertaintyAgent → (PIs/quantiles, coverage report) → DecisionAgent
- DecisionAgent -> (decisions, simulated outcomes) -> CriticAgent
- CriticAgent → (actionable feedback: swap model, widen PI, adjust constraint/penalty) → orchestrator, then loop continues

Key design choices
------------------
- Explicit agent boundaries: isolates concerns (quality, signal, modeling, uncertainty, decision, evaluation) so each can evolve independently.
- Rolling-origin backtesting: mirrors production deployment with no leakage; all comparisons share identical splits.
- Portfolio mindset: combine a fast, explainable baseline with a stronger ML model; select via backtest metrics.
- Calibrated uncertainty: simple quantile regression plus conformal correction keeps intervals honest under shift.
- Decision-first: optimization respects capacity/budget and embeds risk via service-level penalties.
- Critic closes the loop: uses both forecast metrics and decision regret to drive next iteration, not just accuracy.

Operational flow
----------------
- Inputs: long-form time series with identifiers, timestamps, target `y`, and optional covariates.
- Backtesting: rolling windows generate (train, validate) splits; metrics logged per split and per model.
- Serving: once a model is selected, forecasts + intervals feed the decision policy; simulator estimates cost and constraint adherence.
- Monitoring: coverage, WAPE/sMAPE, regret, and violation counts tracked; CriticAgent raises actions when thresholds breached.

