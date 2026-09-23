"""
src/eda_core.py
===============
Exploratory data analysis on data/interim/cleaned_dataset_final.csv (read-only).

Run (works from any folder):
    python src/eda_core.py
    python src/eda_core.py --data path/to/cleaned_dataset_final.csv --out-dir path/to/eda

Every chart is saved as a PNG in outputs/eda/, plus statistical_tests_summary.csv.

What this dataset is (so the charts are read correctly)
  * Two different sources are stacked: CA (2001-2026) and NY (2019-2026). They use different
    label schemes, so many categories (most treatments, HealthPlan, CoverageType, ...) exist in
    only one state. Charts therefore split or colour by State wherever pooling would mislead.
  * The overturn rate drifts strongly over time and differently per State, so year trends are
    always shown per State.
  * RealAvg*/MatchedHCPCSCount are a category-level LOOKUP (one fixed number per treatment
    label), not case-level money. They are shown only as a "proxy", and NO significance test is
    run on them (such a test would only measure treatment mix).
  * With ~100k rows every chi-square p-value is ~0, so the tests report an effect size
    (Cramer's V) as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from data_cleaning import TRAIN_MAX_YEAR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "cleaned_dataset_final.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "eda"
OUTPUT_DIR = DEFAULT_OUT            # set by run()

STATE_COLORS = {"CA": "#2E86AB", "NY": "#E63946"}       # same meaning in every chart
BOTH_COLOR = "#9AA5B1"                                  # label used by both states
NOT_REPORTED = ("Not_Reported", "Unknown")
AGE_ORDER = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]

REQUIRED_COLUMNS = [
    "CaseID", "State", "Year", "Outcome", "Diagnosis_Primary", "Treatment_Primary", "HealthPlan",
    "CoverageType", "DenialType", "AgeRange", "Gender", "ReviewSpeed_IMRType",
    "DaysToReview", "DaysToAdopt", "DaysToReview_Missing", "DaysToAdopt_Missing",
    "RealAvgMedicarePaid", "RealAvgMedicarePaid_Missing", "Diagnosis_HasMultiple",
]

# -----------------------------------------------------------
# GLOBAL STYLE
# -----------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "font.size": 10.5, "font.family": "DejaVu Sans", "legend.frameon": False,
})

PALETTE = ["#2E86AB", "#E63946", "#F4A261", "#2A9D8F", "#8E44AD", "#457B9D", "#E76F51",
           "#06A77D", "#D62828", "#6A4C93", "#118AB2", "#EF476F", "#FFD166", "#073B4C", "#7209B7"]


# =============================================================
# HELPERS
# =============================================================

def _save(fig, name: str) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(OUTPUT_DIR) / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def _colors(n: int, offset: int = 0) -> list:
    """n distinct colours (curated palette, tab20 families for larger n)."""
    if n <= len(PALETTE):
        base = list(PALETTE)
    else:
        base = []
        for name in ["tab20", "tab20b", "tab20c"]:
            cmap = matplotlib.colormaps[name]
            base.extend(cmap(i) for i in range(cmap.N))
        base = base * (n // len(base) + 1)
    off = offset % len(base)
    return (base[off:] + base[:off])[:n]


def _boxplot(ax, data, labels, **kwargs):
    """boxplot that works on old and new matplotlib ('labels' was renamed 'tick_labels' in 3.9)."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


def _style_boxes(bp, colors) -> None:
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#222222")


def _label_colors(d: pd.DataFrame, col: str, threshold: float = 0.9) -> dict:
    """label -> colour of the State that holds >= threshold of its rows, else the neutral 'both' colour."""
    share = pd.crosstab(d[col], d["State"], normalize="index")
    return {lab: (STATE_COLORS[share.loc[lab].idxmax()] if share.loc[lab].max() >= threshold else BOTH_COLOR)
            for lab in share.index}


def _state_legend(ax, states=("CA", "NY"), extra_handles=None, ncol=3, both=False) -> None:
    handles = [Patch(facecolor=STATE_COLORS[s], label=f"{s} only") for s in states if s in STATE_COLORS]
    if both:
        handles.append(Patch(facecolor=BOTH_COLOR, label="used in both states"))
    handles += extra_handles or []
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=ncol)


def _is_overturned(df: pd.DataFrame) -> pd.Series:
    return (df["Outcome"] == "Overturned").astype(float)


def _period(year: pd.Series) -> pd.Series:
    return pd.Series(np.select([year <= TRAIN_MAX_YEAR, year == TRAIN_MAX_YEAR + 1],
                               ["train", "val"], default="test"), index=year.index)


def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}\n"
                                f"Run data_cleaning.py first, or pass --data <path>.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Cleaned file is missing expected columns: {missing}")
    print(f"Loaded: {path}  Shape: {df.shape}")
    return df


# =============================================================
# 1. TARGET AND DATA STRUCTURE
# =============================================================

def plot_outcome_balance(df: pd.DataFrame) -> None:
    counts = df["Outcome"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(counts.index, counts.values, color=_colors(len(counts)), width=0.55)
    for bar, p, c in zip(bars, pct.values, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{c:,}\n({p}%)", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_title("Outcome Distribution: Upheld vs Overturned")
    ax.set_ylabel("Number of Cases")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.margins(y=0.15)
    _save(fig, "01_outcome_balance.png")


def plot_field_availability(df: pd.DataFrame) -> None:
    """% of cases per State that carry a REAL (not filled-in) value. Shows structural missingness."""
    avail = {
        "HealthPlan": df["HealthPlan"] != "Not_Reported",
        "CoverageType": df["CoverageType"] != "Not_Reported",
        "ReviewSpeed_IMRType": df["ReviewSpeed_IMRType"] != "Not_Reported",
        "DaysToReview": df["DaysToReview_Missing"] == 0,
        "DaysToAdopt": df["DaysToAdopt_Missing"] == 0,
        "Medicare payment proxy": df["RealAvgMedicarePaid_Missing"] == 0,
        "AgeRange": df["AgeRange"] != "Unknown",
        "Gender": df["Gender"] != "Unknown",
    }
    table = pd.DataFrame({k: v.groupby(df["State"]).mean() * 100 for k, v in avail.items()}).T
    table = table[[s for s in ["CA", "NY"] if s in table.columns]]

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="white" if v > 55 else "#222222", fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("% of cases with a real value")
    ax.set_title("Which Fields Exist in Which State")
    fig.tight_layout()
    _save(fig, "02_field_availability_by_state.png")


def plot_proxy_coverage(df: pd.DataFrame, top_n: int = 12) -> None:
    """How much of the data has a Medicare payment proxy at all."""
    has = df["RealAvgMedicarePaid_Missing"] == 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 2.2]})

    ax = axes[0]
    by_state = has.groupby(df["State"]).mean() * 100
    bars = ax.bar(by_state.index, by_state.values, color=[STATE_COLORS[s] for s in by_state.index], width=0.5)
    for bar, v in zip(bars, by_state.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of cases with a proxy value")
    ax.set_title("Proxy Coverage by State")

    ax = axes[1]
    sizes = df["Treatment_Primary"].value_counts().head(top_n).index
    d = df[df["Treatment_Primary"].isin(sizes)]
    cov = has[d.index].groupby(d["Treatment_Primary"]).mean() * 100
    label_color = _label_colors(d, "Treatment_Primary")
    cov = cov.sort_values()
    ax.barh(cov.index, cov.values, color=[label_color[t] for t in cov.index], height=0.65)
    for y, v in enumerate(cov.values):
        ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of cases with a proxy value")
    ax.set_title(f"Proxy Coverage: {top_n} Largest Treatment Categories")
    _state_legend(ax, ncol=3, both=True)
    fig.suptitle("The payment proxy exists only for some treatment labels", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "03_proxy_coverage.png")


def plot_proxy_by_treatment(df: pd.DataFrame, top_n: int = 15) -> None:
    """The lookup itself: one fixed value per treatment label (NOT case-level revenue)."""
    real = df[df["RealAvgMedicarePaid_Missing"] == 0]
    g = (real.groupby(["State", "Treatment_Primary"])["RealAvgMedicarePaid"]
             .agg(val="median", n="size", distinct="nunique").reset_index())
    g = g.sort_values("n", ascending=False).head(top_n).sort_values("val")
    labels = [f"{t} ({s})" for t, s in zip(g["Treatment_Primary"], g["State"])]
    y = np.arange(len(g))
    colors = [STATE_COLORS[s] for s in g["State"]]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(g))))
    ax.hlines(y, 0, g["val"], color=colors, linewidth=2.4, alpha=0.85, zorder=2)
    ax.scatter(g["val"], y, color=colors, s=110, zorder=3, edgecolor="#222222", linewidth=0.9)
    for yy, v, n in zip(y, g["val"], g["n"]):
        ax.text(v + g["val"].max() * 0.03, yy, f"${v:,.2f}  (n={n:,})", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(left=0)
    ax.margins(x=0.25)
    ax.set_xlabel("Average Medicare payment per service ($), category lookup")
    ax.set_title("Medicare Payment Proxy by Treatment Category\n"
                 "(every case in a category carries the same value; this is not case revenue)", fontsize=12)
    _state_legend(ax, ncol=2)
    fig.tight_layout()
    _save(fig, "04_proxy_by_treatment.png")


# =============================================================
# 2. OUTCOME vs CATEGORICAL FEATURES
# =============================================================

def _overturn_rate_bar(df, group_col, title, fname, top_n=None, min_count=30, exclude=(),
                       state_split=False, order=None) -> None:
    """
    Overturn rate per category.
      state_split=True : one bar per State inside each category (categories that exist in both states).
      state_split=False: one bar per category, coloured by the State where the label mostly occurs.
    top_n keeps the N LARGEST categories (not the highest rates, which would favour tiny groups).
    """
    d = df[~df[group_col].isin(exclude)].copy()
    d["_o"] = _is_overturned(d)
    sizes = d.groupby(group_col).size()
    cats = sizes[sizes >= min_count].index
    if top_n:
        cats = sizes[cats].sort_values(ascending=False).head(top_n).index
    d = d[d[group_col].isin(cats)]
    if d.empty:
        print(f"[{group_col}] no category reaches min_count={min_count}; chart skipped.")
        return

    ordered = order is not None
    fig_rows = len(cats)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.27 * fig_rows * (2 if state_split else 1) + 1.5)))

    if state_split:
        g = d.groupby([group_col, "State"])["_o"].agg(total="size", rate="mean").reset_index()
        g["rate"] *= 100
        g = g[g["total"] >= min_count]
        rate = g.pivot(index=group_col, columns="State", values="rate")
        tot = g.pivot(index=group_col, columns="State", values="total")
        if ordered:
            keep = [c for c in order if c in rate.index]
            rate, tot = rate.loc[keep], tot.loc[keep]
        else:
            idx = rate.mean(axis=1).sort_values().index
            rate, tot = rate.loc[idx], tot.loc[idx]
        states = [s for s in ["CA", "NY"] if s in rate.columns]
        h = 0.8 / len(states)
        y = np.arange(len(rate))
        for i, st in enumerate(states):
            pos = y + (i - (len(states) - 1) / 2) * h
            vals = rate[st]
            m = vals.notna().values
            ax.barh(pos[m], vals.values[m], height=h * 0.92, color=STATE_COLORS[st])
            for p, v, n in zip(pos[m], vals.values[m], tot[st].values[m]):
                ax.text(v + 0.8, p, f"{v:.0f}% (n={int(n):,})", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(rate.index.astype(str))
        for st in states:
            share = _is_overturned(df[df["State"] == st]).mean() * 100
            ax.axvline(share, color=STATE_COLORS[st], linestyle="--", linewidth=1.2, alpha=0.8)
        handles = [Patch(facecolor=STATE_COLORS[s],
                         label=f"{s} (dashed = {_is_overturned(df[df['State'] == s]).mean() * 100:.1f}% overall)")
                   for s in states]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
        xmax = np.nanmax(rate.values)
    else:
        g = d.groupby(group_col).agg(total=("_o", "size"), rate=("_o", "mean"))
        g["rate"] *= 100
        label_color = _label_colors(d, group_col)
        g = g.loc[[c for c in order if c in g.index]] if ordered else g.sort_values("rate")
        y = np.arange(len(g))
        ax.barh(y, g["rate"], height=0.68, color=[label_color[c] for c in g.index])
        for yy, r, n in zip(y, g["rate"], g["total"]):
            ax.text(r + 0.8, yy, f"{r:.0f}% (n={n:,})", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(g.index.astype(str))
        overall = _is_overturned(df).mean() * 100
        ax.axvline(overall, color="#333333", linestyle="--", linewidth=1.2, alpha=0.85)
        _state_legend(ax, extra_handles=[plt.Line2D([0], [0], color="#333333", linestyle="--",
                                                    label=f"Overall {overall:.1f}%")], ncol=2, both=True)
        xmax = g["rate"].max()

    if ordered:
        ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.25)
    ax.set_xlabel("Overturn Rate (%)")
    ax.set_title(title)
    _save(fig, fname)


def plot_outcome_vs_categoricals(df: pd.DataFrame) -> None:
    _overturn_rate_bar(df, "Diagnosis_Primary", "Overturn Rate by Diagnosis, per State",
                       "05_overturn_by_diagnosis.png", min_count=50, state_split=True)
    _overturn_rate_bar(df, "Treatment_Primary",
                       "Overturn Rate by Treatment (20 largest categories)\n"
                       "colour = State that uses the label (grey = both)",
                       "06_overturn_by_treatment.png", top_n=20, min_count=100)
    _overturn_rate_bar(df, "HealthPlan", "Overturn Rate by Health Plan (NY, 15 largest)",
                       "07_overturn_by_healthplan.png", top_n=15, min_count=100, exclude=NOT_REPORTED)
    _overturn_rate_bar(df, "CoverageType", "Overturn Rate by Coverage Type (NY)",
                       "08_overturn_by_coveragetype.png", min_count=100, exclude=NOT_REPORTED)
    _overturn_rate_bar(df, "DenialType", "Overturn Rate by Denial Type, per State",
                       "09_overturn_by_denialtype.png", min_count=50, state_split=True)
    _overturn_rate_bar(df, "AgeRange", "Overturn Rate by Age Range, per State",
                       "10_overturn_by_agerange.png", min_count=50, exclude=NOT_REPORTED,
                       state_split=True, order=AGE_ORDER)
    _overturn_rate_bar(df, "Gender", "Overturn Rate by Gender, per State",
                       "11_overturn_by_gender.png", min_count=50, exclude=NOT_REPORTED, state_split=True)
    _overturn_rate_bar(df, "ReviewSpeed_IMRType", "Overturn Rate by Review Type (CA only)",
                       "12_overturn_by_reviewspeed.png", min_count=50, exclude=NOT_REPORTED, state_split=True)


# =============================================================
# 3. TIME TREND (always per State)
# =============================================================

def plot_year_trends(df: pd.DataFrame, min_n: int = 50) -> None:
    """Overturn rate and case volume per Year and State (pooling would mix in NY starting in 2019)."""
    d = df.assign(_o=_is_overturned(df))
    g = d.groupby(["Year", "State"])["_o"].agg(n="size", rate="mean").reset_index()
    g["rate"] *= 100
    years = sorted(df["Year"].unique())
    w = max(8.5, 0.42 * len(years))
    fig, axes = plt.subplots(1, 2, figsize=(w * 2 + 1, 6.5))

    ax = axes[0]
    for st in ["CA", "NY"]:
        s = g[(g["State"] == st) & (g["n"] >= min_n)]
        if s.empty:
            continue
        ax.plot(s["Year"], s["rate"], marker="o", markersize=6, linewidth=2.2, color=STATE_COLORS[st],
                markeredgecolor="white", label=f"{st} (years with n >= {min_n})")
        ax.annotate(f"{s['rate'].iloc[-1]:.0f}%", (s["Year"].iloc[-1], s["rate"].iloc[-1]),
                    textcoords="offset points", xytext=(7, 0), va="center", fontsize=9.5,
                    color=STATE_COLORS[st], fontweight="bold")
    for x, label in [(TRAIN_MAX_YEAR + 0.5, "val"), (TRAIN_MAX_YEAR + 1.5, "test")]:
        ax.axvline(x, color="#888888", linestyle="--", linewidth=1)
        ax.text(x + 0.08, 0.97, label, transform=ax.get_xaxis_transform(), color="#666666", fontsize=9, va="top")
    ax.text(min(years) + 0.1, 0.97, "train  (up to the first dashed line)", transform=ax.get_xaxis_transform(),
            color="#666666", fontsize=9, va="top")
    ax.set_title("Overturn Rate by Year, per State")
    ax.set_xlabel("Year")
    ax.set_ylabel("Overturn Rate (%)")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")
    ax.margins(y=0.15)

    ax = axes[1]
    piv = g.pivot(index="Year", columns="State", values="n").reindex(years).fillna(0)
    bottom = np.zeros(len(piv))
    for st in [s for s in ["CA", "NY"] if s in piv.columns]:
        ax.bar(piv.index, piv[st], bottom=bottom, color=STATE_COLORS[st], label=st, width=0.75)
        bottom += piv[st].values
    ax.set_title("Cases per Year, per State")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of cases")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper left")

    fig.text(0.5, -0.01, "NY data starts in 2019; the latest year may be incomplete.", ha="center",
             fontsize=9, color="#555555")
    fig.subplots_adjust(bottom=0.18, wspace=0.18)
    _save(fig, "13_year_trends_by_state.png")


# =============================================================
# 4. DaysToReview / DaysToAdopt (CA only, real values only)
# =============================================================

def _boxplot_actual_only(df, value_col, missing_flag_col, title, fname, color_offset=0) -> None:
    plot_df = df[df[missing_flag_col] == 0]
    groups = ["Upheld", "Overturned"]
    data = [plot_df.loc[plot_df["Outcome"] == g, value_col].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bp = _boxplot(ax, data, groups, patch_artist=True, widths=0.5,
                  medianprops=dict(color="black", linewidth=1.6),
                  flierprops=dict(marker="o", markersize=3, alpha=0.3, markeredgewidth=0))
    _style_boxes(bp, _colors(len(groups), offset=color_offset))
    states = "/".join(sorted(plot_df["State"].unique()))
    ax.set_title(f"{title}\nreal values only, {states} only "
                 f"(n: Upheld={len(data[0]):,}, Overturned={len(data[1]):,})", fontsize=11)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    _save(fig, fname)


def plot_review_speed_vs_outcome(df: pd.DataFrame) -> None:
    _boxplot_actual_only(df, "DaysToReview", "DaysToReview_Missing", "Days to Review by Outcome",
                         "14_daystoreview_by_outcome.png", color_offset=4)
    _boxplot_actual_only(df, "DaysToAdopt", "DaysToAdopt_Missing", "Days to Adopt by Outcome",
                         "15_daystoadopt_by_outcome.png", color_offset=10)


# =============================================================
# 5. ADVANCED
# =============================================================

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (0 = independent, 1 = one column determines the other)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else float("nan")


def plot_association_matrix(df: pd.DataFrame) -> None:
    """Which columns say (almost) the same thing? High values against State reveal State proxies."""
    cols = [c for c in ["Outcome", "State", "Diagnosis_Primary", "Treatment_Primary", "HealthPlan",
                        "CoverageType", "DenialType", "AgeRange", "Gender", "ReviewSpeed_IMRType"]
            if c in df.columns]
    m = np.eye(len(cols))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            m[i, j] = m[j, i] = cramers_v(df[cols[i]], df[cols[j]])

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if m[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("Cramer's V")
    ax.set_title("Association Between Categorical Columns (Cramer's V)")
    fig.text(0.5, 0.005, "Values near 1: the two columns carry almost the same information "
                         "(e.g. a column that mostly encodes State).", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "16_association_matrix.png")


def plot_diagnosis_treatment_interaction(df: pd.DataFrame, top_diag: int = 12, top_treat: int = 12,
                                         min_count: int = 20) -> None:
    """Overturn rate for the largest Diagnosis x Treatment combinations (CA and NY pooled)."""
    top_d = df["Diagnosis_Primary"].value_counts().head(top_diag).index
    top_t = df["Treatment_Primary"].value_counts().head(top_treat).index
    sub = df[df["Diagnosis_Primary"].isin(top_d) & df["Treatment_Primary"].isin(top_t)]
    count = sub.pivot_table(index="Diagnosis_Primary", columns="Treatment_Primary", values="Outcome", aggfunc="count")
    rate = sub.pivot_table(index="Diagnosis_Primary", columns="Treatment_Primary", values="Outcome",
                           aggfunc=lambda x: (x == "Overturned").mean() * 100)
    rate = rate.where(count >= min_count).reindex(index=top_d, columns=top_t)

    fig, ax = plt.subplots(figsize=(max(10, 0.68 * rate.shape[1]), max(8, 0.5 * rate.shape[0])))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(rate.values), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(rate.shape[1]))
    ax.set_xticklabels(rate.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(rate.shape[0]))
    ax.set_yticklabels(rate.index, fontsize=8.5)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03).set_label("Overturn Rate (%)")
    ax.set_title(f"Overturn Rate: Diagnosis x Treatment (CA + NY pooled; top {top_diag}x{top_treat}, "
                 f"blank = n<{min_count})")
    fig.tight_layout()
    _save(fig, "17_diagnosis_treatment_heatmap.png")


def _grouped_rate_bars(ax, d, group_col, group_labels, min_n=30) -> None:
    """Helper: overturn rate for the groups of `group_col`, one bar per State."""
    d = d.assign(_o=_is_overturned(d))
    g = d.groupby(["State", group_col])["_o"].agg(n="size", rate="mean").reset_index()
    states = [s for s in ["CA", "NY"] if s in g["State"].unique()]
    x = np.arange(len(group_labels))
    w = 0.8 / max(len(states), 1)
    for i, st in enumerate(states):
        vals, ns = [], []
        for key in group_labels:
            r = g[(g["State"] == st) & (g[group_col] == key)]
            ok = len(r) and r["n"].iloc[0] >= min_n
            vals.append(r["rate"].iloc[0] * 100 if ok else np.nan)
            ns.append(int(r["n"].iloc[0]) if len(r) else 0)
        pos = x + (i - (len(states) - 1) / 2) * w
        m = ~np.isnan(vals)
        ax.bar(pos[m], np.array(vals)[m], width=w * 0.92, color=STATE_COLORS[st], label=st)
        for p, v, n, ok in zip(pos, vals, ns, m):
            if ok:
                ax.text(p, v + 1, f"{v:.0f}%\nn={n:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)


def plot_multi_diagnosis_impact(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _grouped_rate_bars(ax, df, "Diagnosis_HasMultiple", [0, 1])
    ax.set_xticklabels(["Single diagnosis", "Multiple diagnoses"])
    ax.set_title("Overturn Rate: Single vs Multiple Diagnoses, per State")
    ax.set_ylabel("Overturn Rate (%)")
    ax.margins(y=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    fig.tight_layout()
    _save(fig, "18_multi_diagnosis_impact.png")


def plot_proxy_availability_vs_outcome(df: pd.DataFrame) -> None:
    """Do cases WITH a payment proxy differ in outcome from cases without? (they come from different labels)"""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    d = df.assign(_proxy=df["RealAvgMedicarePaid_Missing"])
    _grouped_rate_bars(ax, d, "_proxy", [0, 1])
    ax.set_xticklabels(["Proxy available", "No proxy"])
    ax.set_title("Overturn Rate: Cases With vs Without a Payment Proxy, per State")
    ax.set_ylabel("Overturn Rate (%)")
    ax.margins(y=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    fig.tight_layout()
    _save(fig, "19_proxy_availability_vs_outcome.png")


def plot_state_deep_dive(df: pd.DataFrame, top_n: int = 6) -> None:
    """4 panels: rate by State | rate by State and period (drift) | diagnosis mix | treatment mix."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 13.5))
    d = df.assign(_o=_is_overturned(df), _period=_period(df["Year"]))

    ax = axes[0, 0]
    g = d.groupby("State")["_o"].agg(n="size", rate="mean")
    bars = ax.bar(g.index, g["rate"] * 100, color=[STATE_COLORS[s] for s in g.index], width=0.45)
    for bar, r, n in zip(bars, g["rate"] * 100, g["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1, f"{r:.1f}%\n(n={n:,})", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Overturn Rate by State (all years)")
    ax.set_ylabel("Overturn Rate (%)")
    ax.margins(y=0.25)

    ax = axes[0, 1]
    _grouped_rate_bars(ax, d, "_period", ["train", "val", "test"])
    ax.set_xticklabels([f"train (<= {TRAIN_MAX_YEAR})", f"val ({TRAIN_MAX_YEAR + 1})",
                        f"test (>= {TRAIN_MAX_YEAR + 2})"])
    ax.set_title("Overturn Rate by State and Period (the target drifts)")
    ax.set_ylabel("Overturn Rate (%)")
    ax.margins(y=0.25)
    ax.legend(loc="upper left")

    states = ["CA", "NY"]
    for ax, col, label in [(axes[1, 0], "Diagnosis_Primary", "diagnosis"),
                           (axes[1, 1], "Treatment_Primary", "treatment")]:
        top = df[col].value_counts().head(top_n).index
        sub = df[df[col].isin(top)]
        ct = (pd.crosstab(sub[col], sub["State"], normalize="index") * 100).reindex(top)
        x = np.arange(len(ct))
        w = 0.38
        for i, st in enumerate([s for s in states if s in ct.columns]):
            ax.bar(x + (i - 0.5) * w, ct[st], width=w, label=st, color=STATE_COLORS[st])
        ax.set_xticks(x)
        ax.set_xticklabels(ct.index, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{top_n} Largest {label.title()} Categories: State Share")
        ax.set_ylabel(f"Share within {label} (%)")
        ax.legend(title="State", loc="upper right", fontsize=9.5, title_fontsize=9.5)
        ax.margins(y=0.15)

    fig.suptitle("State-wise Deep Dive: CA vs NY", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "20_state_deep_dive.png")


# =============================================================
# 6. STATISTICAL TESTS
# =============================================================

_STAT_RESULTS: list = []
# fields that only exist in one State are tested inside that State only
FIELD_STATE = {"HealthPlan": "NY", "CoverageType": "NY", "ReviewSpeed_IMRType": "CA"}


def _effect_label(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "moderate" if v < 0.5 else "large"


def chi_square_tests(df: pd.DataFrame, min_count: int = 30) -> None:
    """
    Chi-square test of independence of each feature vs Outcome. With ~100k rows p is ~0 for almost
    anything, so the effect size (Cramer's V) is what to read. Run pooled and per State.
    """
    print("\n--- Chi-square: Outcome vs categorical features (read Cramer's V, not p) ---")
    features = ["State", "Diagnosis_Primary", "Treatment_Primary", "DenialType", "AgeRange",
                "Gender", "HealthPlan", "CoverageType", "ReviewSpeed_IMRType"]
    for col in features:
        scopes = [FIELD_STATE[col]] if col in FIELD_STATE else (["ALL"] if col == "State" else ["ALL", "CA", "NY"])
        for scope in scopes:
            sub = df if scope == "ALL" else df[df["State"] == scope]
            sub = sub[~sub[col].isin(NOT_REPORTED)]
            counts = sub[col].value_counts()
            sub = sub[sub[col].isin(counts[counts >= min_count].index)]
            if sub[col].nunique() < 2 or sub["Outcome"].nunique() < 2:
                continue
            ct = pd.crosstab(sub[col], sub["Outcome"])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            v = cramers_v(sub[col], sub["Outcome"])
            _STAT_RESULTS.append({"test": "Chi-square", "scope": scope, "feature": col, "n": len(sub),
                                  "statistic": round(chi2, 2), "p_value": float(f"{p:.3g}"),
                                  "cramers_v": round(v, 4), "effect": _effect_label(v),
                                  "note": f"dof={dof}, categories={sub[col].nunique()}"})
            print(f"  {col:<22} [{scope:<3}] n={len(sub):>7,}  chi2={chi2:>10.1f}  p={p:.3g}  "
                  f"V={v:.3f} ({_effect_label(v)})")


def pointbiserial_review_speed_test(df: pd.DataFrame) -> None:
    """Correlation of DaysToReview / DaysToAdopt (CA only, real values) with Overturned = 1."""
    print("\n--- Point-biserial correlation: review timing vs Outcome (CA, real values only) ---")
    y_all = _is_overturned(df)
    for col, flag in [("DaysToReview", "DaysToReview_Missing"), ("DaysToAdopt", "DaysToAdopt_Missing")]:
        mask = df[flag] == 0
        x, y = df.loc[mask, col], y_all[mask]
        r, p = stats.pointbiserialr(y, x)
        _STAT_RESULTS.append({"test": "Point-biserial", "scope": "CA", "feature": col, "n": len(x),
                              "statistic": round(r, 4), "p_value": float(f"{p:.3g}"),
                              "cramers_v": np.nan, "effect": _effect_label(abs(r)),
                              "note": "correlation r with Overturned=1"})
        print(f"  {col:<14} n={len(x):>7,}  r={r:+.3f}  p={p:.3g}  ({_effect_label(abs(r))})")


def run_all_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    _STAT_RESULTS.clear()
    chi_square_tests(df)
    pointbiserial_review_speed_test(df)
    print("\n(No significance test is run on RealAvg* / payment-proxy columns: they are a category lookup, "
          "so any test would only measure treatment mix.)")
    result = pd.DataFrame(_STAT_RESULTS)
    out = Path(OUTPUT_DIR) / "statistical_tests_summary.csv"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\nSaved statistical test summary: {out}")
    return result


# =============================================================
# MAIN
# =============================================================

def run(data_path=DEFAULT_DATA, out_dir=DEFAULT_OUT) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(out_dir)
    df = load_data(data_path)

    print("\n===== A. TARGET AND DATA STRUCTURE =====")
    plot_outcome_balance(df)
    plot_field_availability(df)
    plot_proxy_coverage(df)
    plot_proxy_by_treatment(df)

    print("\n===== B. OUTCOME vs FEATURES =====")
    plot_outcome_vs_categoricals(df)

    print("\n===== C. TIME AND STATE =====")
    plot_year_trends(df)
    plot_review_speed_vs_outcome(df)
    plot_state_deep_dive(df)

    print("\n===== D. ADVANCED =====")
    plot_association_matrix(df)
    plot_diagnosis_treatment_interaction(df)
    plot_multi_diagnosis_impact(df)
    plot_proxy_availability_vs_outcome(df)

    print("\n===== E. STATISTICAL TESTS =====")
    run_all_statistical_tests(df)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exploratory data analysis (read-only).")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to cleaned_dataset_final.csv")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help="folder for PNGs and the tests CSV")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.out_dir)