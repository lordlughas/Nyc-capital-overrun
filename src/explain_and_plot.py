"""SHAP explanations and report figures.

SHAP values come from LightGBM's exact TreeSHAP (predict(..., pred_contrib=True)),
computed on the latest backtest fold: model trained on bases up to 2024-05, explaining
its predictions for projects as they stood in 2025-05.

Usage: python src/explain_and_plot.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from train_eval import (CATEGORICAL, FEATURES, TARGET, fit_lightgbm, rolling_folds,
                        to_lgb_frame)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"

# Reference palette (light mode)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e6e5e1",
     "surface": "#fcfcfb"}
MODEL_LABEL = {"prior": "Constant (base rate)", "reference_class": "Reference class (agency × phase)",
               "logistic": "Logistic regression", "random_forest": "Random forest",
               "lightgbm": "LightGBM"}
FEATURE_LABEL = {
    "agency": "Managing agency", "phase": "Current phase", "category": "Ten-year plan category",
    "budget_line_prefix": "Budget line prefix", "borough": "Borough",
    "log_budget": "Budget size (log10 $)", "spend_pct": "Share of budget spent",
    "n_subprojects": "Number of sub-projects", "has_schedule": "Has schedule data",
    "months_since_design_start": "Months since design start",
    "months_since_constr_start": "Months since construction start",
    "months_in_phase": "Months in current phase",
    "months_to_forecast_completion": "Months to forecast completion",
    "schedule_slip_4m": "Schedule slip, last 4 months",
    "budget_change_4m": "Budget change, last 4 months",
    "spend_change_4m": "Spend change, last 4 months",
    "growth_since_first_record": "Budget growth since first record",
    "months_since_first_record": "Months since first budget record",
    "n_hist_budget_changes": "Past budget revisions (count)",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True,
        "grid.color": C["grid"], "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    })


def label(f):
    if f.startswith("kw_"):
        return "Name mentions: " + f[3:].replace("_", "/")
    return FEATURE_LABEL.get(f, f)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figures

def fig_overrun_by_phase(df):
    order = ["Pre-Design", "Pending", "Design", "Procurement", "Construction", "Close-out"]
    g = df.groupby("phase")[TARGET].agg(["mean", "size"]).reindex(order)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(range(len(g)), g["mean"] * 100, color=C["blue"], width=0.6)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m * 100 + 0.6, f"{m:.0%}", ha="center", color=C["ink"], fontsize=9)
        ax.text(i, -2.6, f"n={n:,}", ha="center", color=C["muted"], fontsize=7.5)
    ax.set_xticks(range(len(g)), order)
    ax.set_ylim(-4, max(g["mean"]) * 100 + 5)
    ax.set_ylabel("% with >10% budget growth in 12 months")
    ax.grid(axis="x", visible=False)
    ax.set_title("Overrun risk peaks between design and procurement")
    save(fig, "fig1_overrun_by_phase.png")


def fig_model_comparison(metrics):
    models = ["reference_class", "logistic", "random_forest", "lightgbm"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for ax, key, base, title in [
        (axes[0], "roc_auc", 0.5, "ROC-AUC"),
        (axes[1], "pr_auc", metrics["prior"]["base_rate"], "PR-AUC (average precision)")]:
        for i, m in enumerate(models):
            v = metrics[m][key]
            lo, hi = metrics[m][key + "_ci"]
            col = C["blue"] if m == "lightgbm" else C["muted"]
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(v, i, "o", color=col, ms=8, mec=C["surface"], mew=2)
            ax.text(hi + 0.006, i, f"{v:.3f}", va="center", fontsize=8.5, color=C["ink"])
        ax.axvline(base, color=C["ink2"], ls="--", lw=1)
        ax.text(base, -0.75, " chance" if key == "roc_auc" else " base rate",
                fontsize=7.5, color=C["ink2"], va="center")
        ax.set_ylim(-1.0, len(models) - 0.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
    axes[1].set_xlim(left=0.1)
    fig.suptitle("Pooled test performance with 95% project-cluster bootstrap CIs",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=C["ink"], y=1.04)
    save(fig, "fig2_model_comparison.png")


def fig_curves(pred):
    models = [("reference_class", C["orange"]), ("logistic", C["aqua"]), ("lightgbm", C["blue"])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for m, col in models:
        fpr, tpr, _ = roc_curve(pred[TARGET], pred[m])
        axes[0].plot(fpr, tpr, color=col, label=MODEL_LABEL[m])
        pr, rc, _ = precision_recall_curve(pred[TARGET], pred[m])
        axes[1].plot(rc, pr, color=col, label=MODEL_LABEL[m])
    axes[0].plot([0, 1], [0, 1], ls="--", color=C["muted"], lw=1)
    axes[1].axhline(pred[TARGET].mean(), ls="--", color=C["muted"], lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", ylim=(0, 0.8))
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "fig3_roc_pr.png")


def fig_lift_and_calibration(pred):
    p = pred.copy()
    p["decile"] = pd.qcut(p["lightgbm"].rank(method="first"), 10, labels=False)
    g = p.groupby("decile").agg(obs=(TARGET, "mean"), pred=("lightgbm", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    ax = axes[0]
    ax.bar(g.index + 1, g["obs"] * 100, color=C["blue"], width=0.65)
    ax.axhline(p[TARGET].mean() * 100, ls="--", color=C["ink2"], lw=1)
    ax.text(0.6, p[TARGET].mean() * 100 + 0.8, "overall rate", fontsize=7.5, color=C["ink2"])
    for i in [1, 10]:
        ax.text(i, g.loc[i - 1, "obs"] * 100 + 0.8, f"{g.loc[i - 1, 'obs']:.0%}",
                ha="center", fontsize=8.5, color=C["ink"])
    ax.set(xlabel="Risk decile (1 = lowest predicted risk)", ylabel="Observed overrun rate (%)",
           title="Observed overruns by risk decile")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    lim = max(g["obs"].max(), g["pred"].max()) * 100 + 4
    ax.plot([0, lim], [0, lim], ls="--", color=C["muted"], lw=1)
    ax.plot(g["pred"] * 100, g["obs"] * 100, "-o", color=C["blue"], ms=6,
            mec=C["surface"], mew=1.5)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk (%)",
           ylabel="Observed overrun rate (%)", title="Calibration (LightGBM)")
    save(fig, "fig4_lift_calibration.png")


def fig_shap(contrib: pd.DataFrame, X: pd.DataFrame):
    imp = contrib.abs().mean().sort_values()
    top = imp.tail(15)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(range(len(top)), top.values, color=C["blue"], height=0.6)
    ax.set_yticks(range(len(top)), [label(f) for f in top.index])
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What drives predicted overrun risk")
    save(fig, "fig5_shap_importance.png")

    numeric_top = [f for f in imp.sort_values(ascending=False).index
                   if f not in CATEGORICAL and not f.startswith("kw_")][:4]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6))
    for ax, f in zip(axes.ravel(), numeric_top):
        x = X[f].astype(float)
        ok = x.notna()
        lo, hi = np.nanpercentile(x, [1, 99])
        ok &= x.between(lo, hi)
        ax.scatter(x[ok], contrib.loc[ok, f], s=6, color=C["blue"], alpha=0.25, lw=0)
        ax.axhline(0, color=C["ink2"], lw=0.8)
        ax.set(xlabel=label(f), ylabel="SHAP (log-odds)")
        ax.set_title(label(f), fontsize=9.5)
    fig.suptitle("How the top numeric features move the prediction", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    save(fig, "fig6_shap_dependence.png")

    # Categorical effects: mean SHAP by level for agency and phase
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, f in zip(axes, ["phase", "agency"]):
        s = pd.DataFrame({"lvl": X[f].astype(str), "v": contrib[f]})
        g = s.groupby("lvl")["v"].agg(["mean", "size"])
        g = g[g["size"] >= 60].sort_values("mean")
        if f == "agency":
            g = pd.concat([g.head(5), g.tail(5)])
        cols = [C["orange"] if v > 0 else C["aqua"] for v in g["mean"]]
        ax.barh(range(len(g)), g["mean"], color=cols, height=0.6)
        ax.set_yticks(range(len(g)), g.index)
        ax.axvline(0, color=C["ink2"], lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="Mean SHAP (log-odds)", title=label(f))
    fig.suptitle("Category effects (orange raises risk, teal lowers it)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=C["ink"], y=1.02)
    fig.tight_layout()
    save(fig, "fig7_shap_categories.png")
    return imp.sort_values(ascending=False)


# ---------------------------------------------------------------- main

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    df = pd.read_parquet(ROOT / "data" / "processed" / "modelling_table.parquet")
    pred = pd.read_parquet(RESULTS / "backtest_predictions.parquet")
    metrics = json.loads((RESULTS / "metrics.json").read_text())

    fig_overrun_by_phase(df)
    fig_model_comparison(metrics["pooled"])
    fig_curves(pred)
    fig_lift_and_calibration(pred)

    test_period, train_periods = list(rolling_folds(df))[-1]
    tr = df[df.base_period.isin(train_periods)]
    te = df[df.base_period == test_period]
    model, cats, _ = fit_lightgbm(tr[FEATURES], tr[TARGET], tr.fms_id)
    Xte = to_lgb_frame(te, cats)
    contrib = pd.DataFrame(model.predict(Xte, pred_contrib=True)[:, :-1],
                           columns=FEATURES, index=te.index)
    imp = fig_shap(contrib, te[FEATURES])
    imp.round(4).to_csv(RESULTS / "shap_importance.csv", header=["mean_abs_shap"])

    # Decile summary for the report
    p = pred.copy()
    p["decile"] = pd.qcut(p["lightgbm"].rank(method="first"), 10, labels=False) + 1
    dec = p.groupby("decile").agg(observed=(TARGET, "mean"), predicted=("lightgbm", "mean"),
                                  n=(TARGET, "size")).round(3)
    dec.to_csv(RESULTS / "risk_deciles.csv")
    print(imp.head(12).round(3).to_string())
    print(dec.to_string())


if __name__ == "__main__":
    main()
