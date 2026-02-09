Trade-offs and Alternatives
===========================

Scope vs realism
----------------
- Chosen: lean scaffolding with clear agent seams; minimal dependencies (pandas, numpy, scikit-learn, xgboost, streamlit).
- Alternative: full feature store + workflow orchestration (Airflow/Flyte); skipped to keep footprint small.

Model portfolio
---------------
- Chosen: seasonal naive + gradient-boosted regression with calendar/event features; fast to train, strong baselines.
- Alternatives: Prophet, SARIMA, TFT/DeepAR; not included to avoid heavy dependencies and GPU needs.

Uncertainty
-----------
- Chosen: quantile regression + conformal adjustment; easy to calibrate and explain.
- Alternatives: Bayesian methods, ensembles, bootstrap; richer but slower and harder to monitor.

Decision layer
--------------
- Chosen: constrained allocation with risk penalty and simple simulator; good for capacity/inventory-style decisions.
- Alternatives: stochastic programming or full RL; more expressive but heavier and data-hungry.

Evaluation
----------
- Chosen: rolling-origin backtests, WAPE/sMAPE/MAE, coverage, regret vs oracle, constraint violations.
- Alternatives: MASE/MAPE variants per hierarchy, probabilistic scores (CRPS); can be added in `evaluation.py`.

Orchestration
-------------
- Chosen: lightweight custom router with explicit handoffs.
- Alternatives: true LangChain or microservice agents, adopted later once interfaces stabilize.

