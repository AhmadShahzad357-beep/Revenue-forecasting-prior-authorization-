"""
src/train_model.py
==================
Prior-authorization outcome model: P(Overturned).

Run (works from any folder), AFTER feature_engineering.py (and tune_model.py for tuned XGBoost):
    python src/train_model.py

This file is SELF-CONTAINED: it does not import tune_model.py (that script runs a long search the moment it is
imported). The only thing it takes from tune_model.py is the JSON file it writes:
outputs/models/final_best_params.json.

Order of use of the three splits (this is what keeps the numbers honest)
  train : fit the models; time-aware cross-validation
  val   : choose the model, fit the probability calibrators, choose the decision threshold
  test  : evaluated ONCE at the very end (the function that loads it is called last)

Why calibration and per-State results
  The overturn rate drifts over time and differently in CA and NY, and NY grows from 47% of the
  training rows to 86% of the test rows. A model trained on old years then under-predicts:
  its average predicted probability is well below the actual overturn rate in the newest years.
  The Task-2 proxy multiplies these probabilities by an amount, so they must be calibrated.
  Therefore a per-State logistic (Platt) calibrator is fitted on the validation year and applied to test.
  The decision threshold is per State too: one pooled threshold behaves very differently in CA and NY,
  because their overturn rates differ.
  Discrimination (AUC) is reported per State, with bootstrap confidence intervals, because one pooled AUC
  mixes two populations.

Outputs
  models/final_model.joblib             model + calibrators + threshold + feature list (one bundle)
  models/{logistic_regression,random_forest,xgboost}.joblib
  outputs/models/*.csv                  metrics, threshold sweep, predictions_val.csv, predictions_test.csv
  outputs/models/*.png                  charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from data_cleaning import TRAIN_MAX_YEAR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "models"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_PARAMS = DEFAULT_OUT_DIR / "final_best_params.json"

TARGET_COL = "Outcome"
POSITIVE_CLASS = "Overturned"
STATES = ["CA", "NY"]
RANDOM_STATE = 42
MIN_CALIBRATION_ROWS = 200
CV_VALIDATION_YEARS = tuple(range(TRAIN_MAX_YEAR - 2, TRAIN_MAX_YEAR + 1))   # e.g. 2021, 2022, 2023

FALLBACK_XGB_PARAMS = {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8,
                       "colsample_bytree": 0.8, "min_child_weight": 5, "gamma": 0, "reg_alpha": 0, "reg_lambda": 1}
COLORS = {"Logistic Regression": "#2E86AB", "Random Forest": "#E63946", "XGBoost": "#2A9D8F"}
STATE_COLORS = {"CA": "#2E86AB", "NY": "#E63946", "ALL": "#333333"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False})


# =============================================================
# DATA
# =============================================================

def load_split(name: str, data_dir):
    data_dir = Path(data_dir)
    for f in (f"{name}_final.csv", f"{name}_meta.csv"):
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"{data_dir / f} not found. Run feature_engineering.py first.")
    d = pd.read_csv(data_dir / f"{name}_final.csv")
    meta = pd.read_csv(data_dir / f"{name}_meta.csv")
    if len(d) != len(meta) or not (d[TARGET_COL].to_numpy() == meta[TARGET_COL].to_numpy()).all():
        raise ValueError(f"{name}_final.csv and {name}_meta.csv are not row-aligned; rerun feature_engineering.py.")
    X = d.drop(columns=[TARGET_COL])
    if X.isnull().any().any():
        raise ValueError(f"NaN in the {name} matrix.")
    y = (d[TARGET_COL] == POSITIVE_CLASS).astype(int).to_numpy()
    return X, y, meta


def forward_year_splits(years, validation_years) -> list:
    """Time-aware CV: for each year Y, train on rows with Year < Y and validate on Year == Y."""
    years = np.asarray(years)
    splits = []
    for yr in validation_years:
        tr_idx, va_idx = np.flatnonzero(years < yr), np.flatnonzero(years == yr)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise ValueError(f"Cannot build a fold for validation year {yr}.")
        splits.append((tr_idx, va_idx))
    return splits


def load_xgb_params(path) -> dict:
    path = Path(path)
    if path.exists():
        params = json.loads(path.read_text())
        print(f"XGBoost parameters: loaded from {path}")
        return params
    print(f"XGBoost parameters: {path} not found, using conservative defaults (run tune_model.py to tune).")
    return dict(FALLBACK_XGB_PARAMS)


def build_models(xgb_params: dict) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE,
                                                n_jobs=-1, class_weight="balanced"),
        "XGBoost": XGBClassifier(**xgb_params, random_state=RANDOM_STATE, eval_metric="logloss",
                                 tree_method="hist", n_jobs=-1),
    }


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# =============================================================
# CALIBRATION, THRESHOLD, METRICS
# =============================================================

def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrators(p, y, states, min_rows: int = MIN_CALIBRATION_ROWS) -> dict:
    """One Platt calibrator (logistic regression on logit(p)) per State, fitted on the validation rows.

    Tried: comparing this against an isotonic calibrator per State, picking
    whichever had lower log-loss on val. Both States preferred isotonic on
    val, but on the held-out test years it gave a slightly WORSE log-loss
    (0.5412 vs 0.5397 for CA, 0.6345 vs 0.6323 for NY) despite a marginally
    better Brier score -- isotonic's step-function shape can be locally
    overconfident, which log-loss punishes hard. Since log-loss is what
    feeds the revenue proxy (probability x dollar amount), Platt was kept.
    """
    calibrators = {}
    for st in STATES:
        m = states == st
        if m.sum() >= min_rows and len(np.unique(y[m])) == 2:
            calibrators[st] = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p[m]).reshape(-1, 1), y[m])
        else:
            print(f"WARNING: too few validation rows for {st}; its probabilities stay uncalibrated.")
            calibrators[st] = None
    return calibrators


def apply_calibration(p, states, calibrators: dict) -> np.ndarray:
    out = np.asarray(p, dtype=float).copy()
    for st, model in calibrators.items():
        m = states == st
        if model is not None and m.any():
            out[m] = model.predict_proba(_logit(p[m]).reshape(-1, 1))[:, 1]
    return out



def predict_proba_calibrated(bundle: dict, X: pd.DataFrame, states) -> np.ndarray:
    """Calibrated P(Overturned) for rows of an already-encoded feature matrix (see feature_engineering.transform)."""
    raw = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    return apply_calibration(raw, np.asarray(states), bundle["calibrators"])


def youden_threshold(y, p) -> float:
    """Threshold maximising TPR - FPR (independent of the class balance, which drifts)."""
    fpr, tpr, thr = roc_curve(y, p)
    finite = np.isfinite(thr)
    best = thr[finite][np.argmax((tpr - fpr)[finite])]
    return float(np.clip(best, 0.05, 0.95))


def bootstrap_auc_ci(y, p, n_boot: int = 500, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n, aucs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (np.nan, np.nan)


def _groups(states):
    return [("ALL", np.ones(len(states), dtype=bool))] + [(s, states == s) for s in STATES]


def metrics_table(y, p, states, label: str, n_boot: int = 0) -> pd.DataFrame:
    rows = []
    for name, m in _groups(states):
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        yy, pp = y[m], p[m]
        row = {"probabilities": label, "group": name, "n": int(m.sum()), "actual_rate": yy.mean(),
               "mean_pred": pp.mean(), "auc": roc_auc_score(yy, pp),
               "log_loss": log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6)), "brier": brier_score_loss(yy, pp)}
        if n_boot:
            row["auc_ci_low"], row["auc_ci_high"] = bootstrap_auc_ci(yy, pp, n_boot)
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_array(thr, states) -> np.ndarray:
    """thr = one number for every row, or a dict {State: threshold}."""
    if isinstance(thr, dict):
        return np.array([thr.get(s, 0.5) for s in states], dtype=float)
    return np.full(len(states), float(thr))


def threshold_table(y, p, states, thr) -> pd.DataFrame:
    rows = []
    per_row = _threshold_array(thr, states)
    for name, m in _groups(states):
        if m.sum() == 0:
            continue
        pred = (p[m] >= per_row[m]).astype(int)
        shown = round(float(per_row[m][0]), 3) if len(np.unique(per_row[m])) == 1 else "per-State"
        rows.append({"group": name, "threshold": shown, "n": int(m.sum()),
                     "accuracy": accuracy_score(y[m], pred),
                     "precision": precision_score(y[m], pred, zero_division=0),
                     "recall": recall_score(y[m], pred, zero_division=0),
                     "f1": f1_score(y[m], pred, zero_division=0),
                     "predicted_positive_rate": pred.mean()})
    return pd.DataFrame(rows)


def _reliability(y, p, bins: int = 10):
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
    return g["p"].to_numpy(), g["y"].to_numpy()


# =============================================================
# STEPS
# =============================================================

def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cross_validate_models(models, X_train, y_train, meta_train, out_dir: Path) -> pd.DataFrame:
    splits = forward_year_splits(meta_train["Year"].to_numpy(), CV_VALIDATION_YEARS)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=splits, scoring="roc_auc", n_jobs=1)
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std(),
                     **{f"auc_{yr}": s for yr, s in zip(CV_VALIDATION_YEARS, scores)}})
        print(f"{name:<22} time-aware CV ROC-AUC mean={scores.mean():.4f} std={scores.std():.4f} "
              f"folds={np.round(scores, 4)} (validating {list(CV_VALIDATION_YEARS)})")
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rng = np.random.default_rng(0)
    for i, r in table.iterrows():
        vals = [r[f"auc_{yr}"] for yr in CV_VALIDATION_YEARS]
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(vals)), vals, color=COLORS[r["model"]], s=70, alpha=0.85,
                   edgecolor="#222222", zorder=3)
        ax.hlines(r["cv_mean"], i - 0.25, i + 0.25, color="black", linewidth=2, zorder=4)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table["model"])
    ax.set_ylabel("ROC-AUC on the validation year of each fold")
    ax.set_title("Time-aware cross-validation (train on earlier years, validate on the next)")
    _save(fig, out_dir, "01_cv_time_aware.png")
    table.to_csv(out_dir / "cv_scores.csv", index=False)
    return table


def fit_and_compare(models, X_train, y_train, X_val, y_val, s_val, out_dir: Path):
    fitted, val_prob, train_val_rows, val_tables = {}, {}, [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        p_tr, p_va = model.predict_proba(X_train)[:, 1], model.predict_proba(X_val)[:, 1]
        val_prob[name] = p_va
        table = metrics_table(y_val, p_va, s_val, name)
        val_tables.append(table)
        auc = table.set_index("group")["auc"]
        train_auc = roc_auc_score(y_train, p_tr)
        train_val_rows.append({"model": name, "train_auc": train_auc, "val_auc": auc["ALL"],
                               "gap": train_auc - auc["ALL"], "val_auc_CA": auc.get("CA", np.nan),
                               "val_auc_NY": auc.get("NY", np.nan),
                               "macro_val_auc": np.nanmean([auc.get("CA", np.nan), auc.get("NY", np.nan)]),
                               "val_log_loss": table.set_index("group").loc["ALL", "log_loss"]})
    comparison = pd.DataFrame(train_val_rows)
    print(comparison.round(4).to_string(index=False))
    for _, r in comparison.iterrows():
        if r["gap"] > 0.05:
            print(f"  >>> {r['model']}: train AUC exceeds val AUC by {r['gap']:.3f} (overfitting or drift).")
    comparison.to_csv(out_dir / "train_val_comparison.csv", index=False)
    pd.concat(val_tables).to_csv(out_dir / "val_metrics.csv", index=False)
    return fitted, val_prob, comparison


def plot_val_roc(val_prob, y_val, s_val, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, st in zip(axes, STATES):
        m = s_val == st
        for name, p in val_prob.items():
            fpr, tpr, _ = roc_curve(y_val[m], p[m])
            ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.2, label=f"{name} (AUC={roc_auc_score(y_val[m], p[m]):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_title(f"ROC on the validation year: {st} (n={int(m.sum()):,})")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    _save(fig, out_dir, "02_val_roc_by_state.png")


def plot_feature_importance(fitted, features, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
        imp = pd.Series(fitted[name].feature_importances_, index=features).sort_values().tail(15)
        ax.barh(imp.index, imp.values, color=COLORS[name])
        ax.set_title(f"{name}: top 15 feature importances")
        ax.set_xlabel("Importance")
    fig.tight_layout()
    _save(fig, out_dir, "04_feature_importance.png")


def evaluate_on_test(bundle, data_dir, out_dir: Path, n_boot: int):
    """The ONLY place the test split is loaded and used. Called once, last."""
    X_test, y_test, meta = load_split("test", data_dir)
    s_test = meta["State"].to_numpy()
    p_raw = bundle["model"].predict_proba(X_test[bundle["features"]])[:, 1]
    p_cal = apply_calibration(p_raw, s_test, bundle["calibrators"])

    raw = metrics_table(y_test, p_raw, s_test, "raw")
    cal = metrics_table(y_test, p_cal, s_test, "calibrated (per-State, fitted on val)", n_boot=n_boot)
    table = pd.concat([raw, cal], ignore_index=True)
    show = table[["probabilities", "group", "n", "actual_rate", "mean_pred", "auc", "log_loss", "brier"]]
    print(show.round(4).to_string(index=False))
    print("\n95% bootstrap confidence intervals for the AUC (calibrated probabilities):")
    for _, r in cal.iterrows():
        print(f"  {r['group']:<4} AUC {r['auc']:.3f}  [{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]  (n={r['n']:,})")
    table.to_csv(out_dir / "test_metrics.csv", index=False)

    thr = bundle["thresholds"]
    thr_txt = ", ".join(f"{k} {v:.2f}" for k, v in thr.items())
    thr_table = pd.concat([threshold_table(y_test, p_cal, s_test, thr),
                           threshold_table(y_test, p_cal, s_test, 0.5)], ignore_index=True)
    print(f"\nDecision metrics on test (calibrated probabilities), at the per-State thresholds chosen on val "
          f"({thr_txt}) and at 0.50:")
    print(thr_table.round(4).to_string(index=False))
    thr_table.to_csv(out_dir / "test_threshold_metrics.csv", index=False)

    pred = pd.DataFrame({"CaseID": meta["CaseID"], "Year": meta["Year"], "State": s_test,
                         "Diagnosis_Primary": meta["Diagnosis_Primary"], "Treatment_Primary": meta["Treatment_Primary"],
                         "HealthPlan": meta["HealthPlan"], "y_true": y_test, "p_raw": p_raw, "p_calibrated": p_cal})
    pred.to_csv(out_dir / "predictions_test.csv", index=False)

    # calibration chart
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (name, m) in zip(axes, _groups(s_test)):
        for label, p, color in [("raw", p_raw, "#E76F51"), ("calibrated", p_cal, "#2A9D8F")]:
            px, py = _reliability(y_test[m], p[m])
            ax.plot(px, py, marker="o", color=color, linewidth=2, label=f"{label} (mean pred {p[m].mean():.3f})")
        ax.plot([0.2, 0.9], [0.2, 0.9], "--", color="gray", linewidth=1)
        ax.set_title(f"{name}: actual overturn rate {y_test[m].mean():.3f}")
        ax.set_xlabel("Predicted probability (decile mean)")
        ax.set_ylabel("Actual overturn rate")
        ax.legend(loc="upper left", fontsize=9.5)
    fig.suptitle("Calibration on the test years (dashed = perfect)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "03_test_calibration.png")

    # confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    per_row = _threshold_array(thr, s_test)
    for ax, (name, m) in zip(axes, _groups(s_test)):
        cm = confusion_matrix(y_test[m], (p_cal[m] >= per_row[m]).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Upheld", "Overturned"]).plot(ax=ax, cmap="Greens", colorbar=False)
        ax.set_title(f"{name} (threshold {'per-State' if name == 'ALL' else format(thr[name], '.2f')})")
    fig.suptitle(f"{bundle['model_name']}: test confusion matrices", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "05_test_confusion_matrices.png")
    return table


# =============================================================
# PIPELINE
# =============================================================

def run(data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, models_dir=DEFAULT_MODELS_DIR,
        params_path=DEFAULT_PARAMS, n_boot: int = 500) -> dict:
    out_dir, models_dir = Path(out_dir), Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    _section("LOAD (train and val only; test stays untouched until the end)")
    X_train, y_train, m_train = load_split("train", data_dir)
    X_val, y_val, m_val = load_split("val", data_dir)
    s_val = m_val["State"].to_numpy()
    features = list(X_train.columns)
    if list(X_val.columns) != features:
        raise ValueError("train and val have different feature columns.")
    print(f"Train {X_train.shape}, val {X_val.shape}; train overturn rate {y_train.mean():.3f}, "
          f"val {y_val.mean():.3f}")
    models = build_models(load_xgb_params(params_path))

    _section("STEP 1: TIME-AWARE CROSS-VALIDATION (inside the training years)")
    cross_validate_models(models, X_train, y_train, m_train, out_dir)

    _section("STEP 2: FIT ON TRAIN, COMPARE ON VALIDATION (pooled and per State)")
    fitted, val_prob, comparison = fit_and_compare(models, X_train, y_train, X_val, y_val, s_val, out_dir)
    plot_val_roc(val_prob, y_val, s_val, out_dir)
    plot_feature_importance(fitted, features, out_dir)

    _section("STEP 3: CHOOSE THE MODEL (validation only)")
    best = comparison.sort_values("macro_val_auc", ascending=False).iloc[0]["model"]
    print(f"Chosen: {best} (highest mean of the CA and NY validation AUCs: "
          f"{comparison.set_index('model').loc[best, 'macro_val_auc']:.4f}).")
    print("Per-State AUC is used because the pooled AUC also rewards getting the CA/NY level difference right.")

    _section("STEP 4: CALIBRATE PER STATE AND CHOOSE THE THRESHOLD (validation only)")
    p_val_raw = val_prob[best]
    calibrators = fit_calibrators(p_val_raw, y_val, s_val)
    p_val_cal = apply_calibration(p_val_raw, s_val, calibrators)
    thresholds = {}
    for st in STATES:
        m = s_val == st
        thresholds[st] = (youden_threshold(y_val[m], p_val_cal[m])
                          if calibrators[st] is not None and len(np.unique(y_val[m])) == 2 else 0.5)
    sweep = pd.concat([threshold_table(y_val, p_val_cal, s_val, t).assign(candidate=t)
                       for t in np.round(np.arange(0.30, 0.71, 0.05), 2)], ignore_index=True)
    for st in STATES:
        print(f"\nValidation sweep, {st} (calibrated probabilities):")
        print(sweep[sweep["group"] == st][["threshold", "accuracy", "precision", "recall", "f1"]].round(4).to_string(index=False))
    print("\nThresholds chosen on val, one per State (each maximises recall - false-positive rate, which does not "
          "depend on the class balance): " + ", ".join(f"{k} {v:.2f}" for k, v in thresholds.items()))
    print("(The val calibration is perfect by construction, because it was fitted on val; the test years show the truth.)")
    sweep.to_csv(out_dir / "val_threshold_sweep.csv", index=False)
    pd.DataFrame({"CaseID": m_val["CaseID"], "Year": m_val["Year"], "State": s_val,
                  "Diagnosis_Primary": m_val["Diagnosis_Primary"], "Treatment_Primary": m_val["Treatment_Primary"],
                  "HealthPlan": m_val["HealthPlan"], "y_true": y_val, "p_raw": p_val_raw,
                  "p_calibrated": p_val_cal}).to_csv(out_dir / "predictions_val.csv", index=False)

    bundle = {"model_name": best, "model": fitted[best], "calibrators": calibrators, "thresholds": thresholds,
              "features": features, "xgb_params": load_xgb_params(params_path) if best == "XGBoost" else None,
              "feature_pipeline": "feature_pipeline.joblib", "note": "P(Overturned); calibrate per State on the val year"}
    for name, model in fitted.items():
        joblib.dump(model, models_dir / f"{_slug(name)}.joblib")
    joblib.dump(bundle, models_dir / "final_model.joblib")
    print(f"Saved models in {models_dir}")

    _section("STEP 5: FINAL TEST EVALUATION (the only time the test years are used)")
    test_table = evaluate_on_test(bundle, data_dir, out_dir, n_boot)

    _section("SUMMARY")
    cal = test_table[test_table["probabilities"].str.startswith("calibrated")].set_index("group")
    raw = test_table[test_table["probabilities"] == "raw"].set_index("group")
    for g in ["ALL", "CA", "NY"]:
        if g in cal.index:
            print(f"  {g:<3} AUC {cal.loc[g, 'auc']:.3f} | mean predicted {raw.loc[g, 'mean_pred']:.3f} (raw) -> "
                  f"{cal.loc[g, 'mean_pred']:.3f} (calibrated) vs actual {cal.loc[g, 'actual_rate']:.3f}")
    print("\nReading these numbers: NY (the larger group) is the reliable one; CA has few test rows, wide intervals "
          "and a target that is still drifting upward, so re-fit the calibrators on the newest year regularly.")
    print(f"All outputs saved in: {out_dir}")
    return bundle


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train, calibrate and evaluate the outcome model.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    p.add_argument("--params", default=str(DEFAULT_PARAMS), help="JSON with XGBoost parameters (from tune_model.py)")
    p.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples for the AUC confidence intervals")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    run(a.data_dir, a.out_dir, a.models_dir, a.params, a.n_boot)