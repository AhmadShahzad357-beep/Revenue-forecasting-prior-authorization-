"""
src/tune_model.py
=================
XGBoost hyper-parameter tuning with TIME-AWARE cross-validation.

Run (works from any folder), AFTER feature_engineering.py:
    python src/tune_model.py                    # full search (can take a while)
    python src/tune_model.py --n-iter 30        # cheaper search

Why time-aware CV
  The data is split by time (train <= TRAIN_MAX_YEAR, val next year, test after), and the overturn
  rate drifts over the years. Shuffled K-fold would let the model "see the future" while tuning and
  choose settings that look better than they are. Here every fold trains on earlier years and
  validates on ONE later year (forward chaining), exactly like the real use.

Stages (the fitted models never see val/test):
  STAGE 1  random search over a wide grid, folds = last 3 training years
  STAGE 2  the 10 best candidates re-scored on 4 folds (a different, larger set of validation years)
  STAGE 3  small exhaustive grid around the best candidate (one step up/down per tuned parameter)
If stage 3 improves the score by less than 0.001, more tuning is unlikely to matter. That is a stopping
rule, not a proof: the model is judged on the validation set in train_model.py.

Outputs (outputs/models/): xgb_tuning_stage{1,2,3}_*.csv, final_best_params.json, 4 charts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier

from data_cleaning import TRAIN_MAX_YEAR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "models"
TARGET_COL = "Outcome"
POSITIVE_CLASS = "Overturned"
RANDOM_STATE = 42

STAGE1_YEARS = tuple(range(TRAIN_MAX_YEAR - 2, TRAIN_MAX_YEAR + 1))   # e.g. 2021, 2022, 2023
STAGE2_YEARS = tuple(range(TRAIN_MAX_YEAR - 3, TRAIN_MAX_YEAR + 1))   # e.g. 2020 .. 2023

PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.1, 0.15],
    "n_estimators": [150, 200, 300, 400, 500, 600],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 9],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5, 0.7],
    "reg_alpha": [0, 0.05, 0.1, 0.5, 1],
    "reg_lambda": [0.5, 1, 1.5, 2, 3],
}

PALETTE = ["#2E86AB", "#E63946", "#F4A261", "#2A9D8F", "#8E44AD", "#457B9D", "#E76F51"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False})


# =============================================================
# DATA + FOLDS
# =============================================================

def load_train(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    for name in ("train_final.csv", "train_meta.csv"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"{data_dir / name} not found. Run feature_engineering.py first.")
    train = pd.read_csv(data_dir / "train_final.csv")
    meta = pd.read_csv(data_dir / "train_meta.csv")
    if len(train) != len(meta) or not (train[TARGET_COL].to_numpy() == meta[TARGET_COL].to_numpy()).all():
        raise ValueError("train_final.csv and train_meta.csv are not row-aligned; rerun feature_engineering.py.")
    X = train.drop(columns=[TARGET_COL])
    if X.isnull().any().any():
        raise ValueError("NaN in the training matrix.")
    y = (train[TARGET_COL] == POSITIVE_CLASS).astype(int).to_numpy()
    return X, y, meta


def forward_year_splits(years, validation_years) -> list:
    """Forward chaining: for each year Y, train on rows with Year < Y and validate on Year == Y."""
    years = np.asarray(years)
    splits = []
    for yr in validation_years:
        tr_idx, va_idx = np.flatnonzero(years < yr), np.flatnonzero(years == yr)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise ValueError(f"Cannot build a fold for validation year {yr}.")
        splits.append((tr_idx, va_idx))
    return splits


def describe_folds(splits, meta: pd.DataFrame, validation_years) -> None:
    for (tr_idx, va_idx), yr in zip(splits, validation_years):
        ny = (meta["State"].to_numpy()[va_idx] == "NY").mean()
        print(f"  fold validating {yr}: train rows={len(tr_idx):,}  validation rows={len(va_idx):,} (NY {ny:.0%})")


def _model(**params) -> XGBClassifier:
    # n_jobs=1: the search itself runs in parallel, so nested parallelism would only slow things down
    return XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss",
                         tree_method="hist", n_jobs=1)


def _neighbors(value, options):
    """The tried value and its neighbours (one step up / down) in the option list."""
    if value not in options:
        return [value]
    i = options.index(value)
    return options[max(0, i - 1): min(len(options), i + 2)]


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# =============================================================
# TUNING
# =============================================================

def run(data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, n_iter: int = 100, top_k: int = 10,
        refine_params=("max_depth", "learning_rate", "n_estimators"), n_jobs: int = -1) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, meta = load_train(data_dir)
    years = meta["Year"].to_numpy()
    print(f"Train matrix: {X.shape}, overturn rate {y.mean():.3f}")

    # ---------------- STAGE 1 ----------------
    splits1 = forward_year_splits(years, STAGE1_YEARS)
    print(f"\nSTAGE 1: random search, {n_iter} candidates x {len(splits1)} time-aware folds")
    describe_folds(splits1, meta, STAGE1_YEARS)
    search = RandomizedSearchCV(_model(), PARAM_DISTRIBUTIONS, n_iter=n_iter, scoring="roc_auc", cv=splits1,
                                n_jobs=n_jobs, random_state=RANDOM_STATE, refit=False, verbose=1)
    t0 = time.time()
    search.fit(X, y)
    print(f"Stage 1 done in {(time.time() - t0) / 60:.1f} minutes")
    stage1 = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    stage1.to_csv(out_dir / "xgb_tuning_stage1_broad.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stage1["mean_test_score"], bins=20, color=PALETTE[0], edgecolor="white", alpha=0.9)
    ax.axvline(stage1["mean_test_score"].max(), color=PALETTE[1], linestyle="--", linewidth=2,
               label=f"Best: {stage1['mean_test_score'].max():.4f}")
    ax.set_title(f"Stage 1: CV ROC-AUC of {n_iter} random candidates (time-aware folds)")
    ax.set_xlabel("mean ROC-AUC over the folds")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, out_dir, "07_stage1_score_distribution.png")

    numeric_params = ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "min_child_weight"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, p in zip(axes.flat, numeric_params):
        ax.scatter(stage1[f"param_{p}"].astype(float), stage1["mean_test_score"], alpha=0.55,
                   color=PALETTE[2], s=35, edgecolor="#333333", linewidth=0.3)
        ax.set_title(p)
        ax.set_xlabel(p)
        ax.set_ylabel("CV ROC-AUC")
    fig.suptitle("Stage 1: hyper-parameter vs score", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "08_stage1_param_vs_score.png")

    # ---------------- STAGE 2 ----------------
    splits2 = forward_year_splits(years, STAGE2_YEARS)
    print(f"\nSTAGE 2: re-scoring the top {top_k} candidates on {len(splits2)} folds (years {STAGE2_YEARS})")
    describe_folds(splits2, meta, STAGE2_YEARS)
    rows = []
    t0 = time.time()
    for params in stage1.head(top_k)["params"].tolist():
        scores = cross_val_score(_model(**params), X, y, cv=splits2, scoring="roc_auc", n_jobs=n_jobs)
        rows.append({"params": params, "cv_mean": float(scores.mean()), "cv_std": float(scores.std())})
        print(f"  ROC-AUC mean={scores.mean():.4f} std={scores.std():.4f}  {params}")
    print(f"Stage 2 done in {(time.time() - t0) / 60:.1f} minutes")
    stage2 = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    stage2.to_csv(out_dir / "xgb_tuning_stage2_confirmed.csv", index=False)
    best2 = stage2.iloc[0]["params"]

    fig, ax = plt.subplots(figsize=(9, 6))
    pos = np.arange(len(stage2))
    ax.barh(pos, stage2["cv_mean"], xerr=stage2["cv_std"], capsize=4, edgecolor="#222222", linewidth=0.6,
            color=[PALETTE[3] if i == 0 else PALETTE[0] for i in range(len(stage2))])
    ax.set_yticks(pos)
    ax.set_yticklabels([f"Candidate {i + 1}" for i in range(len(stage2))])
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (mean +/- std over the folds)")
    ax.set_title("Stage 2: top candidates re-scored")
    for i, (m, s) in enumerate(zip(stage2["cv_mean"], stage2["cv_std"])):
        ax.text(m + s + 0.001, i, f"{m:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "09_stage2_top_candidates.png")

    # ---------------- STAGE 3 ----------------
    grid = {k: [v] for k, v in best2.items()}
    for k in refine_params:
        grid[k] = _neighbors(best2[k], PARAM_DISTRIBUTIONS[k])
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"\nSTAGE 3: local grid around the best candidate: {n_combos} combinations x {len(splits2)} folds")
    gs = GridSearchCV(_model(), grid, scoring="roc_auc", cv=splits2, n_jobs=n_jobs, refit=False, verbose=1)
    t0 = time.time()
    gs.fit(X, y)
    print(f"Stage 3 done in {(time.time() - t0) / 60:.1f} minutes")
    pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").to_csv(out_dir / "xgb_tuning_stage3_refined.csv", index=False)

    stage2_score = float(stage2.iloc[0]["cv_mean"])
    final_score = float(gs.best_score_)
    final_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
    improvement = final_score - stage2_score

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    scores = [float(stage1["mean_test_score"].max()), stage2_score, final_score]
    bars = ax.bar(["Stage 1\n(3 folds)", "Stage 2\n(4 folds)", "Stage 3\n(4 folds)"], scores,
                  color=[PALETTE[0], PALETTE[2], PALETTE[3]], width=0.5)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005, f"{s:.4f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(min(scores) - 0.01, max(scores) + 0.01)
    ax.set_ylabel("ROC-AUC (time-aware CV)")
    ax.set_title(f"Tuning progress (stage 3 vs stage 2: {improvement:+.4f}).\n"
                 "Stage 1 used different folds, so compare stages 2 and 3 only.", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "10_tuning_progress.png")

    print("\n" + "=" * 70)
    print("FINAL BEST PARAMETERS")
    print("=" * 70)
    for k, v in final_params.items():
        print(f"  {k}: {v}")
    print(f"\nTime-aware CV ROC-AUC (stage 3): {final_score:.4f}   improvement over stage 2: {improvement:+.4f}")
    if improvement < 0.001:
        print("Stage 3 gained < 0.001: further tuning of these parameters is unlikely to matter.")
    else:
        print("Stage 3 still improved the score: one more local round around these parameters could help.")
    print("Note: CV here is time-aware and therefore lower than a shuffled-CV score would be; that is the honest number.")

    with open(out_dir / "final_best_params.json", "w") as f:
        json.dump(final_params, f, indent=2)
    print(f"\nSaved: {out_dir / 'final_best_params.json'}\nNext: run train_model.py (it loads these parameters).")
    return final_params


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-aware XGBoost tuning.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n-iter", type=int, default=100, help="stage-1 random candidates")
    p.add_argument("--top-k", type=int, default=10, help="candidates re-scored in stage 2")
    p.add_argument("--refine-params", default="max_depth,learning_rate,n_estimators",
                   help="comma-separated parameters varied in stage 3")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    run(a.data_dir, a.out_dir, a.n_iter, a.top_k, tuple(a.refine_params.split(",")), a.n_jobs)