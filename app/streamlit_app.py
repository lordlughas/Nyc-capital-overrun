"""Interactive overrun-risk demo with per-prediction SHAP explanation.

Run from the repo root:  streamlit run app/streamlit_app.py
Requires models/ (python src/export_model.py) and data/processed/ (python src/build_dataset.py).
"""
from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_eval import CATEGORICAL, FEATURES  # noqa: E402

st.set_page_config(page_title="Capital project overrun risk", layout="wide")


@st.cache_resource
def load():
    model = lgb.Booster(model_file=str(ROOT / "models" / "overrun_lgbm.txt"))
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    forecast = pd.read_csv(ROOT / "results" / "forecast_2026-05_next12m.csv")
    return model, meta, forecast


def calibrate(meta, p):
    c = meta["calibration"]
    return float(np.interp(p, c["x"], c["y"]))


model, meta, forecast = load()
lv = meta["form_levels"]

st.title("Will this capital project's budget grow by more than 10% in the next 12 months?")
st.caption(f"LightGBM trained on {meta['trained_on']['rows']:,} snapshots of "
           f"{meta['trained_on']['projects']:,} NYC capital projects (May 2023 – May 2025 bases). "
           "Rolling-origin backtest ROC-AUC ≈ 0.68.")

left, right = st.columns([1, 1.3])
with left:
    st.subheader("Project as it stands today")
    agency = st.selectbox("Managing agency", lv["agency"], index=lv["agency"].index("DDC"))
    phase = st.selectbox("Current phase", lv["phase"], index=2)
    category = st.selectbox("Ten-year plan category", lv["category"])
    borough = st.selectbox("Borough", lv["borough"])
    budget = st.number_input("Current budget ($)", 250_000, 5_000_000_000, 5_000_000, step=250_000)
    spend = st.slider("Share of budget spent", 0.0, 1.0, 0.10, 0.01)
    age = st.slider("Months since first budget record", 0, 220, 36)
    to_completion = st.slider("Months to forecast completion", -60, 120, 24)
    in_phase = st.slider("Months in current phase", 0, 120, 12)
    chg4 = st.number_input("Budget change over last 4 months (%)", -90.0, 300.0, 0.0)
    growth = st.number_input("Budget growth since first record (%)", -90.0, 1000.0, 10.0)
    revisions = st.number_input("Past budget revisions", 0, 10, 1)

row = {f: np.nan for f in FEATURES}
row.update({
    "agency": agency, "phase": phase, "category": category, "borough": borough,
    "budget_line_prefix": meta["agency_default_prefix"].get(agency, np.nan),
    "log_budget": np.log10(budget), "spend_pct": spend, "months_since_first_record": age,
    "months_to_forecast_completion": to_completion, "months_in_phase": in_phase,
    "budget_change_4m": np.log1p(chg4 / 100), "growth_since_first_record": np.log1p(growth / 100),
    "n_hist_budget_changes": revisions, "n_subprojects": 1, "has_schedule": 1,
})
X = pd.DataFrame([row])[FEATURES]
for c in CATEGORICAL:
    X[c] = pd.Categorical(X[c], categories=meta["categories"][c])
for c in [f for f in FEATURES if f.startswith("kw_")]:
    X[c] = 0

raw = float(model.predict(X)[0])
risk = calibrate(meta, raw)
contrib = pd.Series(model.predict(X, pred_contrib=True)[0][:-1], index=FEATURES)

with right:
    st.subheader("Prediction")
    ref = meta["reference_rates"].get(f"{agency}|{phase}", meta["overall_rate"])
    c1, c2 = st.columns(2)
    c1.metric("Overrun risk (calibrated)", f"{risk:.0%}")
    c2.metric(f"{agency} projects in {phase} (historical)", f"{ref:.0%}")
    st.markdown("**Why:** largest contributions to this prediction (log-odds)")
    top = contrib[contrib.abs() > 0.01].sort_values(key=abs, ascending=False).head(8)
    st.bar_chart(top.rename("SHAP").to_frame(), horizontal=True)

    st.subheader("Highest-risk active projects, May 2026 snapshot")
    show = forecast.head(15)[["project_name", "agency", "phase", "budget", "risk"]].copy()
    show["budget"] = show["budget"].map(lambda v: f"${v / 1e6:,.1f}M")
    show["risk"] = show["risk"].map(lambda v: f"{v:.0%}")
    st.dataframe(show, hide_index=True, width="stretch")
