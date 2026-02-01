import pathlib
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from agentic_forecast.config import SystemConfig  # noqa: E402
from agentic_forecast.orchestrator import run_pipeline  # noqa: E402


st.set_page_config(page_title="Agentic Forecasting + Decision Demo", layout="wide")
st.title("Agentic Forecasting + Decision System")
st.caption("Forecast → Uncertainty → Decision → Critic loop")


def load_data(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


data_path = st.sidebar.text_input(
    "Data CSV path", value=str(ROOT / "sales.csv")
)
horizon = st.sidebar.slider("Horizon (days)", min_value=7, max_value=60, value=14, step=7)

if st.sidebar.button("Run pipeline"):
    try:
        cfg = SystemConfig()
        artifacts = run_pipeline(data_path=data_path, config=cfg, horizon=horizon)

        st.subheader("Backtest metrics (avg)")
        st.json(artifacts["backtest"])

        st.subheader("Forecast and intervals")
        st.line_chart(
            artifacts["forecast"].set_index(cfg.data.time_col)[
                ["forecast", "lower", "upper"]
            ]
        )

        st.subheader("Decision allocations")
        st.dataframe(
            artifacts["decisions"][
                [cfg.data.time_col, "forecast", "allocation", "cost"]
            ]
        )
        st.metric("Simulated mean cost", f"{artifacts['decision_info']['sim_mean_cost']:.2f}")
        st.metric("Stockout rate", f"{artifacts['decision_info']['stockout_rate']:.2%}")

        st.subheader("Critic recommendations")
        for rec in artifacts["critic"]["recommendations"]:
            st.write(f"- {rec}")

    except Exception as e:
        st.error(f"Run failed: {e}")
else:
    st.info("Provide a CSV and press 'Run pipeline' to execute the agent loop.")

