"""
src/feature_engineering.py
==========================
cleaned_dataset_final.csv  ->  train / val / test model matrices.

Run (works from any folder), AFTER data_cleaning.py and check_leakage_columns.py:
    python src/feature_engineering.py

Pipeline: time-based split -> one-hot + target encoding -> scaling -> multicollinearity
report -> feature selection.  EVERYTHING that learns from data (category lists, target-encoding
maps, scaler, feature selection) is fitted on the TRAIN rows only and merely applied to val/test.
The fitted pipeline is saved (models/feature_pipeline.joblib) and reloaded at the end to prove
that applying it to raw val rows reproduces the saved val matrix exactly.

Design choices (from check_leakage_columns.py and the data audit)
  * Features are a WHITELIST (below). A column that is not on it can never slip in by accident;
    every unused column is printed together with the reason.
  * Not used: Year (only the split key: later years are never seen in training), DaysToReview /
    DaysToAdopt (only known after the review, CA-only), the revenue lookup columns (category-level
    numbers, not case-level; kept for the Task-2 proxy), free text, constants.
  * Metadata (CaseID, Year, State, Treatment_Primary, ...) is saved next to each split in
    *_meta.csv, row-aligned, for per-State evaluation and the Task-2 proxy aggregation.

Outputs
  data/processed/{train,val,test}_final.csv   features + Outcome (same format as before)
  data/processed/{train,val,test}_meta.csv    CaseID, Year, State, Diagnosis_Primary, Treatment_Primary,
                                              HealthPlan, Outcome
  outputs/reports/split_summary.csv, vif_report.csv, feature_importance.csv
  models/feature_pipeline.joblib
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from data_cleaning import TRAIN_MAX_YEAR      # single source of truth for the split year

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "cleaned_dataset_final.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_MODELS = PROJECT_ROOT / "models"

VAL_YEAR = TRAIN_MAX_YEAR + 1
TARGET_COL = "Outcome"
POSITIVE_CLASS = "Overturned"

# ---------------- feature whitelist ----------------
ONEHOT_COLS = ["State", "Gender", "DenialType", "CoverageType", "AgeRange", "ReviewSpeed_IMRType"]
TARGETENC_COLS = ["Diagnosis_Primary", "Treatment_Primary", "HealthPlan"]
# Treatment_Count is NOT used: it only takes the values 1 and 2, so Treatment_HasMultiple = Treatment_Count - 1
# exactly (perfectly collinear; the VIF report caught this).
NUMERIC_COLS = ["Diagnosis_Count", "Diagnosis_HasMultiple", "Treatment_HasMultiple"]
CONTINUOUS_COLS = ["Diagnosis_Count"]                              # scaled together with the encodings
META_COLS = ["CaseID", "Year", "State", "Diagnosis_Primary", "Treatment_Primary", "HealthPlan", TARGET_COL]

SMOOTHING = 10          # target-encoding smoothing
N_FOLDS = 5             # out-of-fold target encoding on the training rows
DROP_FRACTION = 0.10    # drop the least important 10% of features (train-only random forest)
RANDOM_STATE = 42

NOT_USED_REASONS = {
    "Year": "split key only: later years never appear in training",
    "DaysToReview": "known only after the review; CA-only (NY rows hold an imputed constant)",
    "DaysToReview_Missing": "marks CA vs NY, not information",
    "DaysToAdopt": "happens after the decision",
    "DaysToAdopt_Missing": "marks CA vs NY, not information",
    "RealAvgSubmittedCharge": "category-level lookup, not case-level (used for the Task-2 proxy only)",
    "RealAvgMedicareAllowed": "category-level lookup, not case-level (used for the Task-2 proxy only)",
    "RealAvgMedicarePaid": "category-level lookup, not case-level (used for the Task-2 proxy only)",
    "MatchedHCPCSCount": "category-level lookup, not case-level",
    "RealAvgSubmittedCharge_Missing": "lookup availability flag", "RealAvgMedicareAllowed_Missing": "lookup availability flag",
    "RealAvgMedicarePaid_Missing": "lookup availability flag", "MatchedHCPCSCount_Missing": "lookup availability flag",
    "Findings_or_Summary": "written after the decision", "Agent": "free text / identifier",
    "References": "free text / identifier", "CaseID": "identifier (kept in the meta file)",
    "DiagnosisCategory": "raw combo string (use Diagnosis_Primary)",
    "TreatmentCategory": "raw combo string (use Treatment_Primary)",
    "DiagnosisCategory_Full": "raw combo string", "TreatmentCategory_Full": "raw combo string",
    "DiagnosisSubCategory": "high-cardinality, state-specific labels", "TreatmentSubCategory": "high-cardinality, state-specific labels",
    "Is_Amended_Group": "constant (no conflicting duplicate CaseIDs)",
    "Treatment_Count": "identical information to Treatment_HasMultiple (only the values 1 and 2 occur)",
    "CMS_Avg_Bene_Avg_Risk_Scre": "State proxy (two values, one per State)",
}


# =============================================================
# LOAD + SPLIT
# =============================================================

def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}\nRun data_cleaning.py first, or pass --data <path>.")
    df = pd.read_csv(path, low_memory=False)
    needed = set(ONEHOT_COLS + TARGETENC_COLS + NUMERIC_COLS + META_COLS)
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Cleaned file is missing expected columns: {missing}")
    used = ONEHOT_COLS + TARGETENC_COLS + NUMERIC_COLS
    nan_cols = [c for c in used + [TARGET_COL, "Year"] if df[c].isnull().any()]
    if nan_cols:
        raise ValueError(f"NaN in model columns (data_cleaning.py should have removed them): {nan_cols}")
    bad = set(df[TARGET_COL].unique()) - {"Upheld", POSITIVE_CLASS}
    if bad:
        raise ValueError(f"Unexpected {TARGET_COL} values: {sorted(bad)}")
    print(f"Loaded: {path}  shape={df.shape}")
    return df


def report_unused_columns(df: pd.DataFrame) -> None:
    used = set(ONEHOT_COLS + TARGETENC_COLS + NUMERIC_COLS + [TARGET_COL])
    unused = [c for c in df.columns if c not in used]
    print(f"Using {len(used) - 1} source columns; NOT used ({len(unused)}):")
    for c in unused:
        print(f"  - {c:<34} {NOT_USED_REASONS.get(c, '(not on the whitelist)')}")


def time_based_split(df: pd.DataFrame):
    train = df[df["Year"] <= TRAIN_MAX_YEAR].copy()
    val = df[df["Year"] == VAL_YEAR].copy()
    test = df[df["Year"] > VAL_YEAR].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One of train/val/test is empty; check TRAIN_MAX_YEAR and the Year column.")
    return train, val, test


def split_summary(train, val, test) -> pd.DataFrame:
    rows = []
    for name, d in [("train", train), ("val", val), ("test", test)]:
        rows.append({"split": name, "years": f"{d['Year'].min()}-{d['Year'].max()}", "rows": len(d),
                     "overturn_rate": round((d[TARGET_COL] == POSITIVE_CLASS).mean(), 4),
                     "NY_share": round((d["State"] == "NY").mean(), 4)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    gap = table.loc[2, "overturn_rate"] - table.loc[0, "overturn_rate"]
    print(f"\nOverturn rate moves {gap:+.3f} from train to test and NY goes from "
          f"{table.loc[0, 'NY_share']:.0%} to {table.loc[2, 'NY_share']:.0%} of the rows: report metrics per State "
          f"and check probability calibration.")
    return table


# =============================================================
# ENCODING (fit on train, apply anywhere)
# =============================================================

def _target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET_COL] == POSITIVE_CLASS).astype(int)


def _dummy_name(col: str, level) -> str:
    """Column name safe for XGBoost / LightGBM (no [ ] < > , : " { })."""
    return re.sub(r'[\[\]<>,:"{}]', "_", f"{col}_{level}")


def fit_artifacts(train: pd.DataFrame) -> dict:
    """Everything learned from the training rows (nothing from val/test)."""
    y = _target(train)
    mu = float(y.mean())
    art = {"train_max_year": TRAIN_MAX_YEAR, "smoothing": SMOOTHING, "target_enc": {}, "onehot_levels": {}}
    for col in ONEHOT_COLS:
        art["onehot_levels"][col] = sorted(train[col].unique().tolist())
    for col in TARGETENC_COLS:
        g = y.groupby(train[col]).agg(["sum", "count"])
        smoothed = (g["sum"] + SMOOTHING * mu) / (g["count"] + SMOOTHING)
        art["target_enc"][col] = {"map": smoothed.to_dict(), "global_mean": mu}
    names = [_dummy_name(c, lvl) for c, lv in art["onehot_levels"].items() for lvl in lv]
    names += [f"{c}_TargetEnc" for c in TARGETENC_COLS] + NUMERIC_COLS
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature names after sanitising dummy names.")
    art["all_features"] = names
    return art


def oof_target_encode(train: pd.DataFrame, y: pd.Series, cols=TARGETENC_COLS,
                      n_splits: int = N_FOLDS, smoothing: int = SMOOTHING, seed: int = RANDOM_STATE) -> dict:
    """
    Out-of-fold target encoding for the TRAINING rows: a row is encoded with statistics computed
    WITHOUT that row's fold (and with that fold-train's own mean as the prior), so no row ever
    sees its own label. Val/test use the full-train maps stored in the artifacts.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for col in cols:
        enc = np.full(len(train), np.nan)
        for tr_idx, ho_idx in kf.split(train):
            fold_y = y.iloc[tr_idx]
            prior = fold_y.mean()
            g = fold_y.groupby(train[col].iloc[tr_idx]).agg(["sum", "count"])
            smoothed = (g["sum"] + smoothing * prior) / (g["count"] + smoothing)
            enc[ho_idx] = train[col].iloc[ho_idx].map(smoothed).fillna(prior).to_numpy()
        out[col] = pd.Series(enc, index=train.index)
    return out


def encode_frame(df: pd.DataFrame, art: dict, te_values: dict | None = None) -> pd.DataFrame:
    """One-hot (fixed train levels) + target encoding + numeric columns. Same code for every split."""
    parts = {}
    for col, levels in art["onehot_levels"].items():
        for lvl in levels:
            parts[_dummy_name(col, lvl)] = (df[col] == lvl).astype("int8")
    for col in TARGETENC_COLS:
        enc = art["target_enc"][col]
        if te_values is not None:
            parts[f"{col}_TargetEnc"] = te_values[col].astype(float)
        else:
            parts[f"{col}_TargetEnc"] = df[col].map(enc["map"]).fillna(enc["global_mean"]).astype(float)
    for col in NUMERIC_COLS:
        parts[col] = df[col].astype(float) if col in CONTINUOUS_COLS else df[col].astype("int8")
    X = pd.DataFrame(parts, index=df.index)
    return X[art["all_features"]]


def unseen_category_report(df: pd.DataFrame, art: dict, name: str) -> None:
    """Rows whose value in a target-encoded or one-hot column never appeared
    in training. Target-encoded columns fall back to the train mean (already
    handled); one-hot columns silently become an all-zero dummy row (no
    warning previously existed for this case), so both are reported here,
    not just the target-encoded ones.
    """
    te_shares = {c: (~df[c].isin(art["target_enc"][c]["map"])).mean() for c in TARGETENC_COLS}
    oh_shares = {c: (~df[c].isin(art["onehot_levels"][c])).mean() for c in ONEHOT_COLS}
    te_txt = ", ".join(f"{c}: {s:.1%}" for c, s in te_shares.items())
    oh_txt = ", ".join(f"{c}: {s:.1%}" for c, s in oh_shares.items())
    print(f"  {name}: unseen category in a target-encoded column (falls back to train mean): {te_txt}")
    print(f"  {name}: unseen category in a one-hot column (becomes an all-zero row): {oh_txt}")


def scale_columns(X: pd.DataFrame, art: dict) -> pd.DataFrame:
    X = X.copy()
    cols = art["scale_cols"]
    X[cols] = art["scaler"].transform(X[cols])
    return X


def transform(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Apply the saved pipeline to any cleaned frame (val/test/new cases)."""
    X = scale_columns(encode_frame(df, art), art)
    return X[art["final_features"]]


# =============================================================
# REPORTS + SELECTION
# =============================================================

def vif_report(X: pd.DataFrame, cols: list) -> pd.DataFrame | None:
    """
    Multicollinearity of the NON-dummy features only (dummies of one column are collinear by
    construction, so their VIF says nothing). Report only, nothing is dropped.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("statsmodels not installed: VIF report skipped (pip install statsmodels).")
        return None
    A = X[cols].to_numpy(dtype=float)
    rows = [{"feature": c, "VIF": float(variance_inflation_factor(A, i))} for i, c in enumerate(cols)]
    table = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    table["high_multicollinearity"] = table["VIF"] > 10
    print(table.round(3).to_string(index=False))
    return table


def select_features(X_train: pd.DataFrame, y_train: pd.Series, drop_fraction: float = DROP_FRACTION):
    """Drop the least important `drop_fraction` of features according to a TRAIN-only random forest."""
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE,
                                n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    imp = (pd.DataFrame({"feature": X_train.columns, "importance": rf.feature_importances_})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    n_drop = max(1, int(len(imp) * drop_fraction))
    dropped = imp.tail(n_drop)["feature"].tolist()
    kept = [c for c in X_train.columns if c not in dropped]
    print(imp.head(15).round(4).to_string(index=False))
    print(f"\nDropped the {n_drop} least important of {len(imp)} features: {dropped}")
    return kept, dropped, imp


# =============================================================
# PIPELINE
# =============================================================

def _save_split(name: str, X: pd.DataFrame, d: pd.DataFrame, out_dir: Path) -> None:
    final = X.copy()
    final[TARGET_COL] = d[TARGET_COL].to_numpy()
    final.to_csv(out_dir / f"{name}_final.csv", index=False)
    d[META_COLS].to_csv(out_dir / f"{name}_meta.csv", index=False)
    print(f"Saved: {out_dir / (name + '_final.csv')}  shape={final.shape}")


def verify_roundtrip(val_raw: pd.DataFrame, pipeline_path: Path, out_dir: Path) -> None:
    """Reload the saved pipeline, transform the raw val rows, compare with the saved val matrix."""
    art = joblib.load(pipeline_path)
    fresh = transform(val_raw, art).reset_index(drop=True)
    saved = pd.read_csv(out_dir / "val_final.csv")
    saved_x = saved.drop(columns=[TARGET_COL])
    if list(fresh.columns) != list(saved_x.columns):
        raise AssertionError("Round-trip check FAILED: column names/order differ.")
    if not np.allclose(fresh.to_numpy(dtype=float), saved_x.to_numpy(dtype=float), atol=1e-9):
        raise AssertionError("Round-trip check FAILED: values differ.")
    if not (saved[TARGET_COL].to_numpy() == val_raw[TARGET_COL].to_numpy()).all():
        raise AssertionError("Round-trip check FAILED: target/row order differs.")
    print("Round-trip check passed: the saved pipeline reproduces val_final.csv exactly.")


def run(data_path=DEFAULT_DATA, out_dir=DEFAULT_OUT, reports_dir=DEFAULT_REPORTS, models_dir=DEFAULT_MODELS) -> None:
    out_dir, reports_dir, models_dir = Path(out_dir), Path(reports_dir), Path(models_dir)
    for d in (out_dir, reports_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    _section("LOAD")
    df = load_data(data_path)
    report_unused_columns(df)

    _section(f"STEP 1: TIME-BASED SPLIT (train <= {TRAIN_MAX_YEAR}, val = {VAL_YEAR}, test > {VAL_YEAR})")
    train, val, test = time_based_split(df)
    summary = split_summary(train, val, test)
    summary.to_csv(reports_dir / "split_summary.csv", index=False)

    _section("STEP 2: ENCODING (fitted on train only)")
    art = fit_artifacts(train)
    y_train = _target(train)
    te_oof = oof_target_encode(train, y_train)
    X_train = encode_frame(train, art, te_values=te_oof)
    X_val, X_test = encode_frame(val, art), encode_frame(test, art)
    print(f"{len(art['all_features'])} features: {sum(len(v) for v in art['onehot_levels'].values())} one-hot "
          f"({', '.join(ONEHOT_COLS)}), {len(TARGETENC_COLS)} target-encoded, {len(NUMERIC_COLS)} numeric.")
    unseen_category_report(val, art, "val ")
    unseen_category_report(test, art, "test")

    _section("STEP 3: SCALING (continuous columns, fitted on train only)")
    art["scale_cols"] = [f"{c}_TargetEnc" for c in TARGETENC_COLS] + CONTINUOUS_COLS
    art["scaler"] = StandardScaler().fit(X_train[art["scale_cols"]])
    X_train, X_val, X_test = (scale_columns(X, art) for X in (X_train, X_val, X_test))
    print(f"Scaled {len(art['scale_cols'])} columns: {art['scale_cols']}")

    _section("STEP 4: MULTICOLLINEARITY (report only, continuous features)")
    vif = vif_report(X_train, art["scale_cols"] + ["Diagnosis_HasMultiple", "Treatment_HasMultiple"])
    if vif is not None:
        vif.to_csv(reports_dir / "vif_report.csv", index=False)

    _section("STEP 5: FEATURE SELECTION (random-forest importance, train only)")
    kept, dropped, importance = select_features(X_train, y_train)
    art["final_features"], art["dropped_low_importance"] = kept, dropped
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    _section("STEP 6: SAVE")
    for name, X, d in [("train", X_train[kept], train), ("val", X_val[kept], val), ("test", X_test[kept], test)]:
        if X.isnull().any().any():
            raise ValueError(f"NaN in the {name} matrix.")
        _save_split(name, X, d, out_dir)
    pipeline_path = models_dir / "feature_pipeline.joblib"
    joblib.dump(art, pipeline_path)
    print(f"Saved: {pipeline_path}")
    print(f"\nFinal feature count: {len(kept)}")
    verify_roundtrip(val, pipeline_path, out_dir)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train/val/test model matrices.")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to cleaned_dataset_final.csv")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help="folder for the processed CSVs")
    p.add_argument("--reports-dir", default=str(DEFAULT_REPORTS), help="folder for the report CSVs")
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS), help="folder for feature_pipeline.joblib")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.out_dir, args.reports_dir, args.models_dir)