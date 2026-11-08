"""Build the modelling table: one row per (project, base snapshot).

Each row describes a project as it stood at a base reporting period t, using only
information available at t, and labels it with what happened to its budget over
the following 12 months (three reporting periods later).

Target
    growth_12m = budget(t+12m) / budget(t) - 1
    overrun    = growth_12m > 10%

Usage: python src/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

OVERRUN_THRESHOLD = 0.10
MIN_BUDGET = 250_000          # ignore very small allocations
HORIZON_STEPS = 3             # 3 reporting periods = 12 months

# Map the many raw phase labels to a small set. Only "active" phases are modelled.
PHASE_MAP = {
    "pre-design": "Pre-Design", "initiation": "Pre-Design", "scope development": "Pre-Design",
    "pending": "Pending",
    "design": "Design",
    "construction procurement": "Procurement",
    "construction": "Construction", "pre-construction phase": "Procurement",
    "close-out": "Close-out", "closeout": "Close-out",
}
ACTIVE_PHASES = {"Pre-Design", "Pending", "Design", "Procurement", "Construction", "Close-out"}

KEYWORDS = {
    "kw_reconstruct": r"reconstr|rehab|renovat|restor",
    "kw_new": r"\bnew\b|construction of",
    "kw_bridge": r"bridge",
    "kw_school": r"school|\bps\b|\bis\b",
    "kw_park": r"park|playground|field|garden",
    "kw_water_sewer": r"sewer|water main|\bwm\b|storm|drainage|wastewater|wwtp|reservoir",
    "kw_street": r"street|road|avenue|\bave\b|highway|plaza|sidewalk",
    "kw_mep": r"hvac|boiler|elevator|electrical|roof|fire alarm|mechanical",
    "kw_library": r"library",
    "kw_site": r"acqui|site\b|land",
}


def to_date(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period.astype("Int64").astype(str), format="%Y%m", errors="coerce")


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.days / 30.44


def load_panel() -> pd.DataFrame:
    """One row per project per reporting period."""
    a = pd.read_csv(RAW / "fb86-vt7u.csv", low_memory=False)
    a["phase_raw"] = a["Current Phase"].fillna("").str.strip("() ").str.lower()
    a["phase"] = a["phase_raw"].map(PHASE_MAP).fillna("Other:" + a["phase_raw"])

    # A project (FMS ID) can be split into several agency sub-projects that repeat
    # the same FMS budget. Keep the row with the largest budget; count sub-projects.
    a = a.sort_values(["Reporting Period", "FMS ID", "Total Budget"],
                      ascending=[True, True, False])
    n_sub = a.groupby(["Reporting Period", "FMS ID"]).size().rename("n_subprojects")
    p = a.drop_duplicates(["Reporting Period", "FMS ID"]).set_index(
        ["Reporting Period", "FMS ID"]).join(n_sub).reset_index()

    for c in ["Current Phase Start", "Forecast Completion", "Actual Design Start",
              "Actual Construction Start", "Forecast Current Phase End"]:
        p[c] = pd.to_datetime(p[c], errors="coerce")
    p["date"] = to_date(p["Reporting Period"])
    return p


def history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Budget track record from the long history table, using only records <= t."""
    h = pd.read_csv(RAW / "qj5n-h5qp.csv")
    h = (h.groupby(["FMS ID", "Year-Month Reported"], as_index=False)["Total Budget"].max()
           .sort_values(["FMS ID", "Year-Month Reported"]))
    h = h[h["Total Budget"] > 0]
    h["changed"] = h.groupby("FMS ID")["Total Budget"].diff().abs().gt(1).astype(int)

    keys = panel[["Reporting Period", "FMS ID"]].drop_duplicates()
    m = keys.merge(h, on="FMS ID", how="left")
    m = m[m["Year-Month Reported"] <= m["Reporting Period"]]
    m = m.sort_values(["FMS ID", "Reporting Period", "Year-Month Reported"])
    g = m.groupby(["Reporting Period", "FMS ID"])
    out = pd.DataFrame({
        "first_budget": g["Total Budget"].first(),
        "first_period": g["Year-Month Reported"].first(),
        "n_hist_changes": g["changed"].sum(),
        "n_hist_records": g.size(),
    }).reset_index()
    out["months_since_first_record"] = months_between(
        to_date(out["Reporting Period"]), to_date(out["first_period"]))
    return out.drop(columns="first_period")


def build(labelled_only: bool = True) -> pd.DataFrame:
    np.seterr(divide="ignore", invalid="ignore")
    p = load_panel()
    periods = sorted(p["Reporting Period"].unique())
    step = {per: i for i, per in enumerate(periods)}
    p["step"] = p["Reporting Period"].map(step)

    # Lagged state from the previous snapshot (4 months earlier)
    prev = p[["FMS ID", "step", "Total Budget", "Spend to Date (%)", "Forecast Completion"]].copy()
    prev["step"] += 1
    prev.columns = ["FMS ID", "step", "budget_prev", "spend_prev", "fc_prev"]
    p = p.merge(prev, on=["FMS ID", "step"], how="left")

    # Target: budget three snapshots (12 months) later
    fut = p[["FMS ID", "step", "Total Budget"]].copy()
    fut["step"] -= HORIZON_STEPS
    fut.columns = ["FMS ID", "step", "budget_12m"]
    p = p.merge(fut, on=["FMS ID", "step"], how="left")

    p = p.merge(history_features(p), on=["Reporting Period", "FMS ID"], how="left")

    B = p["Total Budget"]
    df = pd.DataFrame({
        "fms_id": p["FMS ID"],
        "base_period": p["Reporting Period"],
        "project_name": p["FMS Project Name"],
        # categorical
        "agency": p["Managing Agency"],
        "phase": p["phase"],
        "category": p["Ten Year Plan Category"].fillna("UNKNOWN").str.upper().str.strip(),
        "budget_line_prefix": p["Budget Line"].fillna("NA").str.split("-").str[0].str.upper(),
        "borough": np.where(p["Borough"].fillna("").str.contains(","), "Multiple",
                            p["Borough"].fillna("Unknown")),
        # size and progress
        "log_budget": np.log10(B.clip(lower=1)),
        "spend_pct": p["Spend to Date (%)"].clip(0, 1.5),
        "n_subprojects": p["n_subprojects"],
        "has_schedule": p["Actual Design Start"].notna() | p["Current Phase Start"].notna(),
        # schedule
        "months_since_design_start": months_between(p["date"], p["Actual Design Start"]),
        "months_since_constr_start": months_between(p["date"], p["Actual Construction Start"]),
        "months_in_phase": months_between(p["date"], p["Current Phase Start"]),
        "months_to_forecast_completion": months_between(p["Forecast Completion"], p["date"]),
        "schedule_slip_4m": months_between(p["Forecast Completion"], p["fc_prev"]),
        # budget momentum and track record (all observed at or before t)
        "budget_change_4m": np.log(B / p["budget_prev"]).replace([np.inf, -np.inf], np.nan),
        "spend_change_4m": p["Spend to Date (%)"] - p["spend_prev"],
        "growth_since_first_record": np.log(B / p["first_budget"]).replace([np.inf, -np.inf], np.nan),
        "months_since_first_record": p["months_since_first_record"],
        "n_hist_budget_changes": p["n_hist_changes"],
        # target
        "budget": B,
        "budget_12m": p["budget_12m"],
    })
    name = p["FMS Project Name"].fillna("").str.lower()
    for k, pat in KEYWORDS.items():
        df[k] = name.str.contains(pat, regex=True).astype(int)
    df["has_schedule"] = df["has_schedule"].astype(int)

    # Some date fields hold placeholder years (1900, 2999). Treat implausible spans as missing.
    for c in ["months_since_design_start", "months_since_constr_start", "months_in_phase",
              "months_to_forecast_completion"]:
        df.loc[~df[c].between(-120, 360), c] = np.nan
    df.loc[~df["schedule_slip_4m"].between(-120, 120), "schedule_slip_4m"] = np.nan

    df["growth_12m"] = df["budget_12m"] / df["budget"] - 1
    df["log_growth_12m"] = np.log(df["budget_12m"] / df["budget"])
    df["overrun"] = (df["growth_12m"] > OVERRUN_THRESHOLD).astype(int)

    eligible = df["phase"].isin(ACTIVE_PHASES) & (df["budget"] >= MIN_BUDGET)
    if not labelled_only:
        return df[eligible].reset_index(drop=True)
    return df[eligible & (df["budget_12m"] > 0)].reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(OUT / "modelling_table.parquet", index=False)
    full = build(labelled_only=False)
    current = full[full.base_period == full.base_period.max()]
    current.to_parquet(OUT / "current_portfolio.parquet", index=False)
    print(f"current portfolio ({current.base_period.iloc[0]}): {len(current):,} active projects")
    print(f"rows: {len(df):,}  projects: {df.fms_id.nunique():,}")
    print("rows per base period:", df.base_period.value_counts().sort_index().to_dict())
    print(f"overrun rate (>{OVERRUN_THRESHOLD:.0%} in 12 months): {df.overrun.mean():.3f}")


if __name__ == "__main__":
    main()
