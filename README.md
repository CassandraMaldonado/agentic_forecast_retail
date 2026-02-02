Agentic Forecasting + Decision System
-----

This repository scaffolds a production-style, multi-agent forecasting and decision system suited for demand and resource planning. 

What’s inside
-----
- Multi-agent loop: DataQuality -> Signal -> ModelPortfolio -> Uncertainty -> Decision -> Critic.
- Rolling-origin backtesting with leakage guards.
- Forecast models: seasonal naive baseline + gradient-boosted regression features; quantile/conformal uncertainty.
- Decision layer: capacity/budget-aware allocation simulator with cost/risk trade-offs.
- Streamlit demo to visualize forecast, intervals, and recommended actions.
- Documentation: architecture, trade-offs, and evaluation guidance.

Quick start
-----------
1) Environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Run the Streamlit demo
```
streamlit run app.py
```
3) Run a dry-run pipeline
```
python -m agentic_forecast.orchestrator --data sales.csv --horizon 14
```

Repository layout
-----------------
- `src/agentic_forecast/`: agent definitions and orchestration
  - `data_quality.py`: schema/missingness/anomaly checks
  - `signal.py`: decomposition, regime shift flags, feature generation
  - `models/`: baselines, boosted models, quantile/conformal utilities
  - `model_portfolio.py`: rolling-origin training and model selection
  - `uncertainty.py`: interval calibration and coverage evaluation
  - `decision.py`: constrained allocation + simulator
  - `evaluation.py`: forecast + decision metrics
  - `critic.py`: closes the loop and proposes next iteration
  - `orchestrator.py`: LangGraph-style router wiring the agents
- `app.py`: Streamlit demo to inspect forecasts, intervals, and decisions
- `architecture.md`: agent responsibilities and interaction diagram
- `tradeoffs.md`: design choices and alternatives
- `requirements.txt`: minimal dependencies (pandas, numpy, scikit-learn, xgboost, streamlit)

Data expectations
-----------------
- Tabular time series with columns like: `date`, `item_id`, `store_id`, `y`, plus optional covariates (`price`, `promo`, `event`).
- Use `dates.csv`, `stores.csv`, `categories.csv`, and `sales.csv` as sample inputs; the pipeline expects a single long-form table and performs basic validation and type coercion.

How to extend
-------------
- Add models: implement `BaseForecastModel` in `models/` and register in `ModelPortfolioAgent`.
- New decisions: subclass `DecisionPolicy` with your constraints/objective and plug into `DecisionAgent`.
- Better uncertainty: swap in conformalized quantile regressors or simulation-based intervals in `UncertaintyAgent`.

Testing and evaluation
----------------------
- Use rolling-origin backtests via `ModelPortfolioAgent.backtest`.
- Forecast metrics: WAPE, sMAPE, MAE (see `evaluation.py`).
- Decision metrics: realized cost, regret vs oracle, constraint violations.
- Interval diagnostics: coverage vs nominal α, width efficiency.

Status
------
This is a scaffold with sane defaults. It is designed to be readable, extensible and production-friendly.

