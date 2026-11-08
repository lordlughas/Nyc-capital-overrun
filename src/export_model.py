"""Train the deployment model on all labelled data and export it.

Outputs
    models/overrun_lgbm.txt     LightGBM model (used by the Streamlit app)
    models/model_meta.json      categories, feature order, calibration map
    demo/model.json             compact tree dump + metadata for the browser demo

Calibration: raw scores are mapped to observed overrun rates with isotonic regression
fitted on the out-of-sample backtest predictions, because the raw model is
over-confident in its top decile.

Usage: python src/export_model.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from train_eval import CATEGORICAL, FEATURES, NUMERIC, TARGET, fit_lightgbm, to_lgb_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEMO = ROOT / "demo"

FORM_FEATURES = [  # what the demo lets a user set; everything else is left missing
    "agency", "phase", "category", "borough", "log_budget", "spend_pct",
    "months_since_first_record", "months_to_forecast_completion", "months_in_phase",
    "budget_change_4m", "growth_since_first_record", "n_hist_budget_changes",
]


def compact_tree(node):
    if "leaf_value" in node:
        return {"v": round(node["leaf_value"], 7)}
    out = {"f": node["split_feature"], "l": compact_tree(node["left_child"]),
           "r": compact_tree(node["right_child"]), "dl": bool(node["default_left"]),
           "m": node["missing_type"]}
    if node["decision_type"] == "==":
        out["c"] = [int(x) for x in str(node["threshold"]).split("||")]
    else:
        out["t"] = node["threshold"]
    return out


def main():
    MODELS.mkdir(exist_ok=True)
    DEMO.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "processed" / "modelling_table.parquet")
    pred = pd.read_parquet(ROOT / "results" / "backtest_predictions.parquet")

    model, cats, rounds = fit_lightgbm(df[FEATURES], df[TARGET], df.fms_id)
    model.save_model(MODELS / "overrun_lgbm.txt")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(pred["lightgbm"], pred[TARGET])
    calib = {"x": [float(x) for x in iso.X_thresholds_], "y": [float(y) for y in iso.y_thresholds_]}

    # Default budget-line prefix per agency (the demo doesn't ask for it)
    prefix = df.groupby("agency")["budget_line_prefix"].agg(lambda s: s.mode().iloc[0]).to_dict()
    # Reference-class rate by agency x phase, shown next to the model's answer
    ref = (df.groupby(["agency", "phase"])[TARGET].agg(["mean", "size"])
             .query("size >= 30")["mean"].round(3))
    ref = {f"{a}|{p}": float(v) for (a, p), v in ref.items()}
    # Only offer categories with enough history in the form
    counts = {c: df[c].value_counts() for c in ["agency", "category"]}
    form_levels = {c: sorted(counts[c][counts[c] >= 40].index.tolist()) for c in counts}
    form_levels["phase"] = ["Pre-Design", "Pending", "Design", "Procurement", "Construction", "Close-out"]
    form_levels["borough"] = sorted(df["borough"].value_counts().loc[lambda s: s >= 40].index)

    raw_latest = model.predict(to_lgb_frame(df[df.base_period == df.base_period.max()], cats))

    meta = {
        "features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC,
        "categories": cats, "rounds": rounds, "calibration": calib,
        "agency_default_prefix": prefix, "reference_rates": ref, "form_levels": form_levels,
        "overall_rate": float(df[TARGET].mean()),
        "score_percentiles": [float(np.percentile(raw_latest, q)) for q in range(0, 101)],
        "trained_on": {"rows": int(len(df)), "projects": int(df.fms_id.nunique()),
                       "base_periods": [int(x) for x in sorted(df.base_period.unique())]},
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=1))

    dump = model.dump_model()
    demo = {"trees": [compact_tree(t["tree_structure"]) for t in dump["tree_info"]], **meta}

    # Test vectors so the browser implementation can be checked against Python
    sample = df.sample(300, random_state=1)
    Xs = to_lgb_frame(sample, cats)
    rows = []
    for (_, r), p in zip(Xs.iterrows(), model.predict(Xs)):
        x = []
        for f in FEATURES:
            if f in CATEGORICAL:
                x.append(None if pd.isna(r[f]) else cats[f].index(r[f]))
            else:
                x.append(None if pd.isna(r[f]) else float(r[f]))
        rows.append({"x": x, "p": float(p)})
    (DEMO / "test_vectors.json").write_text(json.dumps(rows))

    # Forward forecast: score every active project in the latest snapshot (outcome unknown)
    cur = pd.read_parquet(ROOT / "data" / "processed" / "current_portfolio.parquet")
    raw = model.predict(to_lgb_frame(cur, cats))
    cur = cur.assign(raw_score=raw, risk=iso.predict(raw))
    cols = ["fms_id", "project_name", "agency", "phase", "category", "borough", "budget",
            "spend_pct", "risk", "raw_score"]
    cur.sort_values("raw_score", ascending=False)[cols].to_csv(
        ROOT / "results" / "forecast_2026-05_next12m.csv", index=False)
    top = cur.sort_values("raw_score", ascending=False).head(12)
    demo["forecast"] = {
        "snapshot": int(cur.base_period.iloc[0]), "n_projects": int(len(cur)),
        "expected_overruns": float(cur["risk"].sum()),
        "top": [{"name": str(r.project_name).title()[:70], "agency": r.agency, "phase": r.phase,
                 "budget": float(r.budget), "spend_pct": float(r.spend_pct), "risk": float(r.risk)}
                for r in top.itertuples()],
    }
    # A few real projects the demo can load as starting points
    shuffled = cur.assign(q=pd.qcut(cur.raw_score.rank(method="first"), 5, labels=False)) \
                  .sample(frac=1, random_state=7)
    shuffled = shuffled[shuffled.has_schedule == 1]
    chosen, used = [], set()
    for q in [4, 3, 2, 1, 0]:          # highest-risk quintile first
        g = shuffled[(shuffled.q == q) & ~shuffled.agency.isin(used)]
        r = g.iloc[0]
        chosen.append(r)
        used.add(r.agency)
    picks = pd.DataFrame(chosen)
    demo["examples"] = []
    for r in picks.itertuples():
        demo["examples"].append({
            "name": str(r.project_name).title()[:60], **{f: (None if pd.isna(getattr(r, f))
                                                           else getattr(r, f)) for f in FORM_FEATURES}})
    print(f"forecast: {len(cur):,} active projects as of {cur.base_period.iloc[0]}, "
          f"expected overruns {cur['risk'].sum():.0f}")
    (DEMO / "model.json").write_text(json.dumps(demo, separators=(",", ":"), default=float))
    print(f"deployment model: {rounds} rounds on {len(df):,} rows; "
          f"demo/model.json {(DEMO / 'model.json').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
