"""
src/check_leakage_columns.py
============================
DIAGNOSTIC ONLY -- trains no production model and changes no data.
Run it AFTER data_cleaning.py, BEFORE feature_engineering.py.

Run (works from any folder):
    python src/check_leakage_columns.py
    python src/check_leakage_columns.py --data path/to/cleaned_dataset_final.csv

Three checks
  1. DaysToReview / DaysToAdopt: which rows have them, how they relate to each other and
     to Outcome (both are CA-only fields; NY rows only carry an imputed constant).
  2. Out-of-time single-feature screen for every candidate feature: fit on Year <= TRAIN_MAX_YEAR,
     score on Year > TRAIN_MAX_YEAR. A single feature with a very high AUC is a leakage red flag.
  3. Split drift: size, overturn rate and State mix of train / val / test.

What this script can and cannot tell you
  * It CAN show that a column is statistically suspicious (very high single-feature AUC).
  * It CANNOT prove a column is available before the decision. That is a business question:
    the model is meant to predict Upheld/Overturned when an appeal is FILED, so any field that
    is only known after the review (DaysToReview, DaysToAdopt, Findings_or_Summary) must stay out,
    even if its AUC is low.

Outputs: outputs/reports/single_feature_screen.csv, outputs/reports/split_drift.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from data_cleaning import TRAIN_MAX_YEAR   # single source of truth for the split year

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "cleaned_dataset_final.csv"
DEFAULT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

TARGET_COL = "Outcome"
POSITIVE_CLASS = "Overturned"

PROCESS_COLS = [("DaysToReview", "DaysToReview_Missing"),
                ("DaysToAdopt", "DaysToAdopt_Missing")]

CATEGORICAL_CANDIDATES = ["State", "Diagnosis_Primary", "Treatment_Primary", "HealthPlan",
                          "CoverageType", "DenialType", "AgeRange", "Gender",
                          "ReviewSpeed_IMRType"]
NUMERIC_CANDIDATES = [("DaysToReview", "DaysToReview_Missing"),
                      ("DaysToAdopt", "DaysToAdopt_Missing"),
                      ("Diagnosis_Count", None), ("Treatment_Count", None), ("Year", None)]

SUSPICIOUS_AUC = 0.70     # a single feature this strong deserves a hard look
STRONG_AUC = 0.60
SMOOTHING = 10            # target-encoding smoothing


# =============================================================
# HELPERS
# =============================================================

def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {path}\nRun data_cleaning.py first, or pass --data <path>.")
    df = pd.read_csv(path, low_memory=False)
    needed = ["Year", "State", TARGET_COL] + CATEGORICAL_CANDIDATES + \
             [c for c, _ in NUMERIC_CANDIDATES] + [f for _, f in NUMERIC_CANDIDATES if f]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Cleaned file is missing expected columns: {missing}")
    print(f"Loaded: {path}  shape={df.shape}")
    df["y"] = (df[TARGET_COL] == POSITIVE_CLASS).astype(int)   # helper column, not saved
    return df


def _safe_auc(y, score) -> float:
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else float("nan")


def _split(df: pd.DataFrame):
    train = df[df["Year"] <= TRAIN_MAX_YEAR]
    later = df[df["Year"] > TRAIN_MAX_YEAR]
    return train, later


# =============================================================
# CHECK 1: DaysToReview / DaysToAdopt
# =============================================================

def check_process_column(df: pd.DataFrame, col: str, flag: str) -> None:
    real = df[df[flag] == 0]
    print(f"\n--- {col} ---")
    print(f"Rows with a REAL value: {len(real):,} of {len(df):,}  (states: {sorted(real['State'].unique())})")
    print("(All other rows carry one imputed constant, i.e. a State marker, not information.)")

    print(f"\n{col} by Outcome (real values only):")
    print(real.groupby(TARGET_COL)[col].describe()[["count", "mean", "25%", "50%", "75%", "max"]].round(2))

    within = df[df["State"].isin(real["State"].unique())]
    print(f"\nShare of real-value-missing rows by Outcome (inside {sorted(real['State'].unique())} only):")
    print(within.groupby(TARGET_COL)[flag].mean().round(4).to_string())
    print("(If this differs a lot between outcomes, the field's availability itself would leak the outcome.)")

    y, x = real["y"].values, real[[col]].values
    raw_auc = _safe_auc(y, real[col].values)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tree_auc = float(cross_val_score(tree, x, y, cv=cv, scoring="roc_auc").mean())
    print(f"\nSingle-feature power on real rows: raw AUC={raw_auc:.4f} "
          f"(direction-free {max(raw_auc, 1 - raw_auc):.4f}) | depth-4 tree, 5-fold CV AUC={tree_auc:.4f}")
    best = max(max(raw_auc, 1 - raw_auc), tree_auc)
    if best > SUSPICIOUS_AUC:
        print("  >>> SUSPICIOUS: very strong on its own -- possible leakage.")
    elif best > 0.58:
        print("  >>> Mild signal -- worth noting, not clearly leakage.")
    else:
        print("  >>> Weak: statistical leakage risk is low (timing risk is judged separately).")


def check_adopt_vs_review_timing(df: pd.DataFrame) -> None:
    print("\n--- DaysToAdopt vs DaysToReview: timing relationship ---")
    both = df[(df["DaysToReview_Missing"] == 0) & (df["DaysToAdopt_Missing"] == 0)]
    print(f"Rows where BOTH are real: {len(both):,}")
    if both.empty:
        print("The two fields are never available together, so their order cannot be compared.")
        return
    print(f"Correlation: {both['DaysToReview'].corr(both['DaysToAdopt']):.3f}")
    pct_greater = (both["DaysToAdopt"] > both["DaysToReview"]).mean() * 100
    print(f"DaysToAdopt > DaysToReview in {pct_greater:.1f}% of rows.")
    if pct_greater > 90:
        print(">>> DaysToAdopt is a LATER step than the review (it happens after the decision).")
        print("    It must be excluded from any model meant to predict at filing time.")


def check_extreme_values(df: pd.DataFrame) -> None:
    print("\n--- Extreme values (real rows) ---")
    for col, flag in PROCESS_COLS:
        actual = df.loc[df[flag] == 0, col]
        n_zero = int((actual == 0).sum())
        print(f"{col}: min={actual.min():.0f}, 1st pct={actual.quantile(0.01):.1f}, "
              f"median={actual.median():.1f}, 99th pct={actual.quantile(0.99):.1f}, max={actual.max():.0f}; "
              f"exactly 0: {n_zero:,} ({n_zero / len(actual) * 100:.2f}%)")


# =============================================================
# CHECK 2: out-of-time single-feature screen
# =============================================================

def _target_encoding_auc(train: pd.DataFrame, later: pd.DataFrame, col: str) -> float:
    mu = train["y"].mean()
    g = train.groupby(col)["y"].agg(["sum", "count"])
    enc = (g["sum"] + SMOOTHING * mu) / (g["count"] + SMOOTHING)
    return _safe_auc(later["y"], later[col].map(enc).fillna(mu))


def _tree_auc(train: pd.DataFrame, later: pd.DataFrame, col: str, flag) -> float:
    tr = train if flag is None else train[train[flag] == 0]
    ev = later if flag is None else later[later[flag] == 0]
    if tr["y"].nunique() < 2 or len(ev) == 0:
        return float("nan")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0)
    tree.fit(tr[[col]], tr["y"])
    return _safe_auc(ev["y"], tree.predict_proba(ev[[col]])[:, 1])


def single_feature_screen(df: pd.DataFrame) -> pd.DataFrame:
    train, later = _split(df)
    print(f"Fit on Year <= {TRAIN_MAX_YEAR} ({len(train):,} rows), score on Year > {TRAIN_MAX_YEAR} "
          f"({len(later):,} rows). AUC 0.50 = no signal.")
    rows = []
    for col in CATEGORICAL_CANDIDATES:
        rows.append({"feature": col, "kind": "categorical (target-encoded)",
                     "out_of_time_auc": _target_encoding_auc(train, later, col)})
    for col, flag in NUMERIC_CANDIDATES:
        note = "numeric (depth-3 tree)" + (", real rows only" if flag else "")
        rows.append({"feature": col, "kind": note, "out_of_time_auc": _tree_auc(train, later, col, flag)})
    table = (pd.DataFrame(rows).sort_values("out_of_time_auc", ascending=False)
             .reset_index(drop=True))
    table["verdict"] = np.select(
        [table["out_of_time_auc"] > SUSPICIOUS_AUC, table["out_of_time_auc"] >= STRONG_AUC],
        ["SUSPICIOUS - investigate", "strong but plausible"], default="weak")
    print(table.round(4).to_string(index=False))
    print("\nNotes: 'Year' scores exactly 0.50 by construction (later years never appear in training),")
    print("which is why Year must not be a model feature. Values below 0.50 mean the relationship")
    print("reversed over time (drift).")
    return table


# =============================================================
# CHECK 3: split drift
# =============================================================

def split_drift_report(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["split"] = np.select([d["Year"] <= TRAIN_MAX_YEAR, d["Year"] == TRAIN_MAX_YEAR + 1],
                           ["train", "val"], default="test")
    overall = (d.groupby("split")
                 .agg(rows=("y", "size"), overturn_rate=("y", "mean"),
                      NY_share=("State", lambda s: (s == "NY").mean()))
                 .reindex(["train", "val", "test"]))
    by_state = (d.groupby(["split", "State"])["y"].agg(rows="size", overturn_rate="mean")
                  .reset_index())
    print(overall.round(3).to_string())
    print("\nBy State:")
    print(by_state.round(3).to_string(index=False))

    gap = overall.loc["test", "overturn_rate"] - overall.loc["train", "overturn_rate"]
    print(f"\nOverturn rate moves from {overall.loc['train', 'overturn_rate']:.3f} (train) to "
          f"{overall.loc['test', 'overturn_rate']:.3f} (test): {gap:+.3f}.")
    print(f"NY is {overall.loc['train', 'NY_share']:.0%} of train but {overall.loc['test', 'NY_share']:.0%} of test.")
    print(">>> The test set is a different population from the training set. Report metrics PER STATE,")
    print("    check probability calibration, and do not read the headline AUC as one number for both states.")
    return by_state.assign(level="by_state").pipe(
        lambda x: pd.concat([overall.reset_index().assign(State="ALL", level="overall"), x], ignore_index=True))


# =============================================================
# MAIN
# =============================================================

def run(data_path=DEFAULT_DATA, reports_dir=DEFAULT_REPORTS) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)

    _section("CHECK 1: DaysToReview / DaysToAdopt")
    for col, flag in PROCESS_COLS:
        check_process_column(df, col, flag)
    check_adopt_vs_review_timing(df)
    check_extreme_values(df)

    _section("CHECK 2: OUT-OF-TIME SINGLE-FEATURE SCREEN (all candidate features)")
    screen = single_feature_screen(df)
    screen.to_csv(reports_dir / "single_feature_screen.csv", index=False)

    _section("CHECK 3: TRAIN / VAL / TEST DRIFT")
    drift = split_drift_report(df)
    drift.to_csv(reports_dir / "split_drift.csv", index=False)

    _section("DECISION GUIDE")
    flagged = screen.loc[screen["verdict"].str.startswith("SUSPICIOUS"), "feature"].tolist()
    print(f"Statistically suspicious features (single-feature AUC > {SUSPICIOUS_AUC}): "
          f"{flagged if flagged else 'none'}")
    print("""
Rules for feature_engineering.py:
  * Always exclude fields that only exist AFTER the decision: DaysToAdopt (proven later than the
    review), Findings_or_Summary, and -- for a 'predict at filing time' model -- DaysToReview.
  * Any feature flagged SUSPICIOUS above: find out how and when it is recorded before using it.
  * Do not use Year as a feature (later years are never seen in training).
  * Compare the measured signal with the ceiling: if no single feature is strong, a moderate model
    AUC is plausible and honest -- a very high AUC would be the thing to distrust.
""")
    print(f"Reports saved in: {reports_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage / drift diagnostics (read-only).")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="path to cleaned_dataset_final.csv")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS), help="folder for the CSV reports")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.reports_dir)