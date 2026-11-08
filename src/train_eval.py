"""Rolling-origin backtest of budget-overrun models.

For each test snapshot T, models are trained only on (project, base) pairs whose
12-month outcome was already known at T. This mimics how an agency would use the
model: train on everything resolved so far, score the current portfolio.

Models
    prior            constant: training overrun rate
    reference_class  smoothed overrun rate by agency x phase (reference class forecasting)
    logistic         one-hot + scaled numeric, L2 logistic regression
    random_forest    same inputs, 500 trees
    lightgbm         gradient-boosted trees with native categorical handling

Usage: python src/train_eval.py
"""
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "modelling_table.parquet"
RESULTS = ROOT / "results"
SEED = 42

CATEGORICAL = ["agency", "phase", "category", "budget_line_prefix", "borough"]
NUMERIC = [
    "log_budget", "spend_pct", "n_subprojects", "has_schedule",
    "months_since_design_start", "months_since_constr_start", "months_in_phase",
    "months_to_forecast_completion", "schedule_slip_4m",
    "budget_change_4m", "spend_change_4m",
    "growth_since_first_record", "months_since_first_record", "n_hist_budget_changes",
    "kw_reconstruct", "kw_new", "kw_bridge", "kw_school", "kw_park", "kw_water_sewer",
    "kw_street", "kw_mep", "kw_library", "kw_site",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "overrun"

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    cat_smooth=20, min_data_per_group=50, verbose=-1, seed=SEED,
)


# ---------------------------------------------------------------- folds

def rolling_folds(df: pd.DataFrame, horizon_steps: int = 3, n_test: int = 3):
    periods = sorted(df["base_period"].unique())
    for j in range(len(periods) - n_test, len(periods)):
        train_periods = [p for i, p in enumerate(periods) if i + horizon_steps <= j]
        yield periods[j], train_periods


# ---------------------------------------------------------------- models

class ReferenceClass:
    """Empirical-Bayes smoothed overrun rate per agency x phase, backing off to agency."""

    def __init__(self, k: float = 30.0):
        self.k = k

    def fit(self, X, y):
        d = X[["agency", "phase"]].assign(y=y.values)
        self.g = y.mean()
        a = d.groupby("agency")["y"].agg(["sum", "count"])
        self.agency = (a["sum"] + self.k * self.g) / (a["count"] + self.k)
        ap = d.groupby(["agency", "phase"])["y"].agg(["sum", "count"]).reset_index()
        prior = ap["agency"].map(self.agency).fillna(self.g)
        ap["rate"] = (ap["sum"] + self.k * prior) / (ap["count"] + self.k)
        self.ap = ap.set_index(["agency", "phase"])["rate"]
        return self

    def predict_proba(self, X):
        idx = pd.MultiIndex.from_frame(X[["agency", "phase"]])
        p = pd.Series(self.ap.reindex(idx).values, index=X.index)
        p = p.fillna(X["agency"].map(self.agency)).fillna(self.g)
        return np.c_[1 - p.values, p.values]


def sklearn_preprocessor():
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                        sparse_output=False)
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def to_lgb_frame(X: pd.DataFrame, categories: dict) -> pd.DataFrame:
    X = X[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype("object"), categories=categories[c])
    return X


def fit_lightgbm(X: pd.DataFrame, y: pd.Series, groups: pd.Series, objective="binary"):
    """Choose the number of boosting rounds by 4-fold CV grouped by project, then refit.

    Grouping matters: the same project appears at several snapshots, so a random or
    time-based inner split would put near-identical rows on both sides and reward
    overfitting.
    """
    categories = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL}
    Xl = to_lgb_frame(X, categories)
    params = dict(LGB_PARAMS, objective=objective)
    folds = list(GroupKFold(n_splits=4).split(Xl, y, groups))
    cv = lgb.cv(params, lgb.Dataset(Xl, y), num_boost_round=2000, folds=folds,
                callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = max(50, len(next(iter(cv.values()))))
    model = lgb.train(params, lgb.Dataset(Xl, y), num_boost_round=rounds)
    return model, categories, rounds


def build_models():
    return {
        "prior": None,
        "reference_class": ReferenceClass(),
        "logistic": Pipeline([("prep", sklearn_preprocessor()),
                              ("clf", LogisticRegression(C=0.3, max_iter=3000))]),
        "random_forest": Pipeline([("prep", sklearn_preprocessor()),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=500, min_samples_leaf=15,
                                       max_features="sqrt", n_jobs=-1, random_state=SEED))]),
        "lightgbm": "lgb",
    }


# ---------------------------------------------------------------- metrics

def precision_at(y, p, frac=0.10):
    n = max(1, int(round(len(p) * frac)))
    top = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[top]))


def metrics(y, p):
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "precision_top10": precision_at(y, p, 0.10),
        "lift_top10": precision_at(y, p, 0.10) / float(y.mean()),
    }


def cluster_bootstrap(pred: pd.DataFrame, model: str, n_boot=500, seed=SEED):
    """95% CI for ROC-AUC and PR-AUC, resampling whole projects (rows of one project are correlated)."""
    rng = np.random.default_rng(seed)
    groups = pred.groupby("fms_id").indices
    keys = np.array(list(groups))
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        y, p = pred[TARGET].values[idx], pred[model].values[idx]
        aucs.append(roc_auc_score(y, p))
        aps.append(average_precision_score(y, p))
    return {"roc_auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "pr_auc_ci": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))]}


# ---------------------------------------------------------------- main

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    preds, fold_rows, rounds_used = [], [], {}

    for test_period, train_periods in rolling_folds(df):
        tr = df[df.base_period.isin(train_periods)]
        te = df[df.base_period == test_period].copy()
        print(f"test {test_period}: train bases {train_periods} -> {len(tr):,} rows, test {len(te):,}")
        for name, model in build_models().items():
            if name == "prior":
                p = np.full(len(te), tr[TARGET].mean())
            elif name == "lightgbm":
                m, cats, rounds = fit_lightgbm(tr[FEATURES], tr[TARGET], tr.fms_id)
                rounds_used[str(test_period)] = rounds
                p = m.predict(to_lgb_frame(te, cats))
            else:
                model.fit(tr[FEATURES], tr[TARGET])
                p = model.predict_proba(te[FEATURES])[:, 1]
            te[name] = p
            fold_rows.append({"test_period": int(test_period), "model": name,
                              **metrics(te[TARGET], p)})

        # Secondary task: size of 12-month budget change (log ratio, clipped), median loss
        yr_tr = tr["log_growth_12m"].clip(-1, 1.5)
        mr, cats_r, _ = fit_lightgbm(tr[FEATURES], yr_tr, tr.fms_id, objective="l1")
        te["reg_pred"] = mr.predict(to_lgb_frame(te, cats_r))
        preds.append(te)

    pred = pd.concat(preds, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    models = list(build_models())

    pooled = {}
    for name in models:
        pooled[name] = metrics(pred[TARGET], pred[name])
        if name != "prior":
            pooled[name].update(cluster_bootstrap(pred, name))

    yr = pred["log_growth_12m"].clip(-1, 1.5)
    regression = {
        "mae_model": float(np.mean(np.abs(yr - pred["reg_pred"]))),
        "mae_no_change_baseline": float(np.mean(np.abs(yr - 0))),
        "spearman": float(spearmanr(yr, pred["reg_pred"]).statistic),
    }

    # Error analysis: where does the best model work, where not?
    by = {}
    for col in ["phase", "agency"]:
        rows = []
        for k, g in pred.groupby(col):
            if len(g) >= 150 and g[TARGET].nunique() == 2:
                rows.append({col: k, "n": len(g), "base_rate": g[TARGET].mean(),
                             "roc_auc": roc_auc_score(g[TARGET], g["lightgbm"]),
                             "pr_auc": average_precision_score(g[TARGET], g["lightgbm"])})
        by[col] = pd.DataFrame(rows).sort_values("n", ascending=False).round(3).to_dict("records")

    out = {"pooled": pooled, "folds": folds.round(4).to_dict("records"),
           "regression": regression, "by_segment": by, "lightgbm_rounds": rounds_used,
           "n_rows": int(len(df)), "n_projects": int(df.fms_id.nunique()),
           "n_test_rows": int(len(pred)), "overrun_rate": float(df[TARGET].mean())}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    pred.to_parquet(RESULTS / "backtest_predictions.parquet", index=False)

    print("\nPooled test metrics (3 rolling folds):")
    print(pd.DataFrame(pooled).T[["roc_auc", "pr_auc", "brier", "precision_top10", "lift_top10"]]
          .round(3).to_string())
    print("\nRegression:", {k: round(v, 3) for k, v in regression.items()})


if __name__ == "__main__":
    main()
