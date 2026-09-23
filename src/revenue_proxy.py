"""
src/revenue_proxy.py
====================
Task 2: a transparent payment PROXY for the appeal outcome model (no second ML model).

Run (works from any folder), AFTER train_model.py:
    python src/revenue_proxy.py

What it does
  For every case with a known payment proxy:
      expected_recoverable = P(Overturned) * proxy_amount      (denial reversed, money paid out)
      expected_lost        = P(Upheld)     * proxy_amount      (denial stands)
  and compares the expected totals with what really happened (labels are known for val/test).

What the proxy amount is
  The dataset has NO case-level dollar amounts. The RealAvg* columns hold one fixed number per
  treatment label (average Medicare payment per service, from CMS data), so every case of a label
  carries the same value. It is a category lookup taken from outputs/reports/revenue_lookup_table.csv
  (written by data_cleaning.py). Consequences that must be stated wherever the numbers are shown:
    * totals are an INDEX of exposure, not real dollars (a service price, not the size of a claim);
    * only some treatment labels have a value; the rest are counted separately, never imputed;
    * CA and NY value the same service slightly differently, so do not compare dollars across States.

Outputs (outputs/reports/): revenue_proxy_by_year_state.csv, revenue_proxy_by_treatment.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_OUT = PROJECT_ROOT / "outputs" / "models"
DEFAULT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

PRED_COLUMNS = ["CaseID", "Year", "State", "Treatment_Primary", "y_true", "p_calibrated"]


def load_lookup(reports_dir=DEFAULT_REPORTS) -> dict:
    """{(State, Treatment_Primary): proxy amount} for groups that have a real, single lookup value."""
    path = Path(reports_dir) / "revenue_lookup_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run data_cleaning.py first.")
    t = pd.read_csv(path)
    t = t[(t["real_rows"] > 0) & (t["distinct_paid_values"] == 1)]
    return {(r.State, r.Treatment_Primary): float(r.paid_min) for r in t.itertuples()}


def load_predictions(models_out=DEFAULT_MODELS_OUT) -> pd.DataFrame:
    """Validation + test predictions (labels known). 'split' tells which years were used to fit the calibrators."""
    models_out = Path(models_out)
    frames = []
    for split in ("val", "test"):
        path = models_out / f"predictions_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run train_model.py first.")
        d = pd.read_csv(path)
        missing = [c for c in PRED_COLUMNS if c not in d.columns]
        if missing:
            raise KeyError(f"{path.name} is missing columns {missing}")
        frames.append(d.assign(split=split))
    return pd.concat(frames, ignore_index=True)


def attach_proxy(pred: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    out = pred.copy()
    keys = list(zip(out["State"], out["Treatment_Primary"]))
    out["proxy_amount"] = [lookup.get(k, np.nan) for k in keys]
    return out


def summarize(pred: pd.DataFrame, by: list) -> pd.DataFrame:
    """Expected vs actual proxy dollars. Only cases WITH a proxy enter the dollar columns."""
    d = pred.copy()
    d["has_proxy"] = d["proxy_amount"].notna()
    amt = d["proxy_amount"].fillna(0.0)
    d["exp_recoverable"] = d["p_calibrated"] * amt
    d["exp_lost"] = (1 - d["p_calibrated"]) * amt
    d["act_recoverable"] = d["y_true"] * amt
    d["act_lost"] = (1 - d["y_true"]) * amt
    g = (d.groupby(by, dropna=False)
           .agg(cases=("CaseID", "size"), cases_with_proxy=("has_proxy", "sum"),
                expected_recoverable=("exp_recoverable", "sum"), expected_lost=("exp_lost", "sum"),
                actual_recoverable=("act_recoverable", "sum"), actual_lost=("act_lost", "sum"))
           .reset_index())
    g["proxy_coverage"] = g["cases_with_proxy"] / g["cases"]
    g["recoverable_error_pct"] = np.where(g["actual_recoverable"] > 0,
                                          (g["expected_recoverable"] - g["actual_recoverable"]) / g["actual_recoverable"] * 100,
                                          np.nan)
    return g


def build_tables(models_out=DEFAULT_MODELS_OUT, reports_dir=DEFAULT_REPORTS) -> dict:
    pred = attach_proxy(load_predictions(models_out), load_lookup(reports_dir))
    by_year_state = summarize(pred, ["split", "Year", "State"]).sort_values(["Year", "State"])
    by_treatment = (summarize(pred[pred["split"] == "test"], ["State", "Treatment_Primary"])
                    .query("cases_with_proxy > 0").sort_values("expected_recoverable", ascending=False))
    overall = summarize(pred, ["split", "State"])
    return {"by_year_state": by_year_state, "by_treatment": by_treatment, "overall": overall}


def run(models_out=DEFAULT_MODELS_OUT, reports_dir=DEFAULT_REPORTS) -> dict:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    t = build_tables(models_out, reports_dir)
    t["by_year_state"].to_csv(reports_dir / "revenue_proxy_by_year_state.csv", index=False)
    t["by_treatment"].to_csv(reports_dir / "revenue_proxy_by_treatment.csv", index=False)

    show = t["overall"].copy()
    for c in ["expected_recoverable", "expected_lost", "actual_recoverable", "actual_lost"]:
        show[c] = show[c].map("{:,.0f}".format)
    show["proxy_coverage"] = (show["proxy_coverage"] * 100).round(1).astype(str) + "%"
    show["recoverable_error_pct"] = show["recoverable_error_pct"].round(1)
    print("Payment PROXY (index of exposure, not real dollars): expected vs actual, cases with a proxy only")
    print(show[["split", "State", "cases", "proxy_coverage", "expected_recoverable", "actual_recoverable",
                "recoverable_error_pct"]].to_string(index=False))
    print("\n'val' years were used to fit the calibrators, so its error is small by construction;")
    print("'test' is the honest check. A negative error means the model expected less than what happened.")
    print(f"\nSaved: {reports_dir / 'revenue_proxy_by_year_state.csv'}\nSaved: {reports_dir / 'revenue_proxy_by_treatment.csv'}")
    return t


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Payment proxy for the appeal outcome model.")
    p.add_argument("--models-out", default=str(DEFAULT_MODELS_OUT), help="folder with predictions_val/test.csv")
    p.add_argument("--reports-dir", default=str(DEFAULT_REPORTS), help="folder with revenue_lookup_table.csv")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    run(a.models_out, a.reports_dir)