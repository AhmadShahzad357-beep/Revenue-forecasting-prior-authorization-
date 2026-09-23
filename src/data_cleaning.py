"""
src/data_cleaning.py
====================
MASTER cleaning pipeline -- the single authoritative cleaning script.
(category_standardization.py is fully merged in here and is no longer needed.)

Run (works from any folder):
    python src/data_cleaning.py
    python src/data_cleaning.py --raw path/to/raw.csv --skip-audit

Input : data/raw/final_TRUE_complete_dataset.csv
Output: data/interim/cleaned_dataset_final.csv
        outputs/reports/cms_column_investigation.csv
        outputs/reports/age_range_mapping.csv
        outputs/reports/category_value_counts.csv
        outputs/reports/revenue_lookup_table.csv

Pipeline order (the order matters):
  1. Load + schema check
  2. Audit (read-only)
  3. Drop exact duplicate rows, resolve duplicate CaseIDs      (raw values)
  4. Normalise text, standardise categories / AgeRange / Gender
  5. Add *_Missing flags                                        (before any fill)
  6. Cap outliers   -> REAL values only, bounds from training years only
  7. Impute         -> statistics from training years only
                       (revenue lookup columns are NOT imputed, see IMPUTE_REVENUE_LOOKUP)
  8. Drop low-variance CMS State-proxy columns
  9. Validate + save

No-leakage rule
---------------
Every statistic learned from the data (outlier bounds, medians) is computed from
rows with Year <= TRAIN_MAX_YEAR only -- the same window that
feature_engineering.time_based_split() uses for training -- and is then applied
to all rows. Keep TRAIN_MAX_YEAR in sync with that function.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================
# CONFIG
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "final_TRUE_complete_dataset.csv"
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_FILENAME = "cleaned_dataset_final.csv"

# Must match feature_engineering.time_based_split(): train = Year <= 2023
TRAIN_MAX_YEAR = 2023

# The four Real*/MatchedHCPCSCount columns are NOT case-level values: in this dataset they
# take only ~17 distinct value-combinations, one per (raw) treatment label -- a hand-built
# category lookup. Filling the ~15% missing rows with a median would copy one category's
# price onto unrelated categories, so by default they are left as NaN (see *_Missing flags
# and outputs/reports/revenue_lookup_table.csv). Set True only to reproduce the old behaviour.
IMPUTE_REVENUE_LOOKUP = False

ID_COL = "CaseID"
TARGET_COL = "Outcome"

PROCESS_DAY_COLS = ["DaysToReview", "DaysToAdopt"]
REVENUE_COLS = ["RealAvgSubmittedCharge", "RealAvgMedicareAllowed",
                "RealAvgMedicarePaid", "MatchedHCPCSCount"]
REVENUE_GROUP_KEYS = ["State", "TreatmentCategory"]

# Fields reported by only one state (structural missingness) -> explicit label
STRUCTURAL_CATEGORICAL = ["DiagnosisSubCategory", "TreatmentSubCategory",
                          "ReviewSpeed_IMRType", "HealthPlan", "CoverageType",
                          "Agent", "References"]

CATEGORICAL_COLS = [
    "State", "DiagnosisCategory", "DiagnosisSubCategory", "TreatmentCategory",
    "TreatmentSubCategory", "HealthPlan", "Outcome", "DenialType", "AgeRange",
    "Gender", "ReviewSpeed_IMRType", "CoverageType",
]
TEXT_COLS = ["Findings_or_Summary", "Agent", "References"]

CMS_PREFIX = "CMS_Avg_"
LOW_VARIANCE_MAX_UNIQUE = 10          # CMS columns with <= this many unique values are dropped
CMS_REFERENCE_COL = "CMS_Avg_Bene_Avg_Risk_Scre"   # the one CMS column that is kept

REQUIRED_COLUMNS = (
    [ID_COL, "Year"] + CATEGORICAL_COLS + TEXT_COLS + PROCESS_DAY_COLS + REVENUE_COLS
)

# =============================================================
# CATEGORY MAPS  (CA short-code / variant -> canonical NY full name)
# Matching is spacing / punctuation / case insensitive (see build_normalized_map).
# Every key must map DIRECTLY to its final value (no chains) -- this is asserted.
# =============================================================

DIAGNOSIS_ABBREV_MAP = {
    'CNS/ Neuromusc Dis':        'Central Nervous System/ Neuromuscular Disorder',
    'Nervous System':            'Central Nervous System/ Neuromuscular Disorder',
    'Cardiac/Circ Problem':      'Cardiac/ Circulatory Problems',
    'Circulatory System':        'Cardiac/ Circulatory Problems',
    'Digest System':             'Digestive System/ Gastrointestinal',
    'Digestive System/ GI':      'Digestive System/ Gastrointestinal',
    'Endo/Metabolic':            'Endocrine/ Metabolic/ Nutritional',
    'Endocrine/Metabolic':       'Endocrine/ Metabolic/ Nutritional',
    'Morbid Obesity':            'Endocrine/ Metabolic/ Nutritional',
    'GU/ Kidney Disorder':       'Genitourinary/ Kidney Disorder',
    'Genitourinary Sys':         'Genitourinary/ Kidney Disorder',
    'Immuno Disorders':          'Immunologic Disorders',
    'Ears/Nose/Throat':          'Ears/ Nose/ Throat',
    'Ear and Mastoid':           'Ears/ Nose/ Throat',
    'Orth/Musculoskeletal':      'Orthopedic/ Musculoskeletal',
    'Musculoskeletal':           'Orthopedic/ Musculoskeletal',
    'Resp System':               'Respiratory System',
    'Neoplasms (Tumor)':         'Cancer',
    'Diseases of Blood':         'Blood Disorder',
    'Blood Related Disord':      'Blood Disorder',
    'Infect/Parasit Dx':         'Infectious Disease',
    'Injury Poison Oth':         'Trauma/ Injuries',
    'Skin Subcutaneous':         'Skin Disorders',
    'Disease Eye Adnexa':        'Vision',
    'Pregnancy Childbirth':      'Pregnancy/ Childbirth',
    'Pregnancy/Childbirth':      'Pregnancy/ Childbirth',
    'OB-GYN/ Pregnancy':         'Pregnancy/ Childbirth',
    'Mental Behav Neur':         'Mental Health',
    'Mental Disorder':           'Mental Health',
    'Autism Spectrum Disorder':  'Autism Spectrum',      # NY name for the CA label 'Autism Spectrum'
    # NOTE: the two older scripts disagreed on this one ('Chronic Pain Syndrome' vs
    # 'Pain Management'). Confirm against the NY label list in category_value_counts.csv.
    'Chron Pain Synd':           'Chronic Pain Syndrome',
    'Hlth Factor/Contact':       'Other',
    'Post Surgical Comp':        'Other',
    'Prevention/Good Hlth':      'Other',
    'Sym/Sign Ab Find':          'Other',
    'Malfor/Deform/Abnor':       'Genetic Diseases',
    'Pediatrics':                'Other',
    'Not Applicable':            'Other',
}

TREATMENT_ABBREV_MAP = {
    'Pharmacy/ Prescription Drugs': 'Pharmacy',
    'Dent/Orthodont Proc':   'Dental/ Orthodontic Procedure',
    'Durable Med Equip':     'DME',
    'Diag Imag & Screen':    'Diagnostic Testing (other than Radiology)',
    'Gen Surg Proc':         'Surgical Services',    # also matches 'Gen Surg proc'
    'Surgery':               'Surgical Services',
    'Cardio-Vasc Proc':      'Surgical Services',
    'Rehab/Svcs SNF Inpt':   'Skilled Nursing Facility',
    'Rehab/ Svc - Outpt':    'Skilled Nursing Facility',
    'Emergency/Urg Care':    'Emergency Care/ Emergency Room',
    'Ortho Proc Serv':       'Orthopedic Proc',
    'Chiropractic Services': 'Chiropractic Care',
    'Path Lab Serv':         'Path Lab Proc',
    'Ob-Gyn Proc':           'OB/ GYN Services',
    'Alternative Tx':        'Other Thera Proc',
    # CA label vs NY label of the same category (each label exists in only one state)
    'Autism Related Tx':     'Autism Related Treatment (including ABA)',
    'Durable Medical Equipment (DME) (including Wearable Defibrilllators)': 'DME',
    'Diabetic Equipment':    'Diabetic Equipment/ Supplies/ Self-Management Education',
    # 'Vision Services' was renamed 'Vision Services/Ophthalmology' in CA (2019); NY says 'Vision Care'
    'Vision Services/Ophthalmology': 'Vision Services',
    'Vision Care':           'Vision Services',
}

HEALTHPLAN_MAP = {
    'Archcare Community Life': 'ArchCare Community Life',
    'Molina Healthcare of New York Inc': 'Molina Healthcare of New York, Inc.',
    'Molina Healthcare of New York, Inc': 'Molina Healthcare of New York, Inc.',
    "Orange-Ulster School District's Health Plan": "Orange-Ulster School Districts' Health Plan",
    'Senior Whole Health of New York Inc': 'Senior Whole Health of New York, Inc',
    'Healthfirst Inc.': 'Healthfirst, Inc.',
    # Same organisation renamed over time (old name and new name never / barely overlap in
    # Year); newer name is used as the canonical one. Remove a line to undo a merge.
    'MVP Health Plan': 'MVP Health Care',                                   # 2019-24 -> 2024-26
    'ElderServe Health, Inc.': 'ElderServe Health, Inc./DBA RiverSpring Health',   # 2019-21 -> 2022-26
    'Univera Community Health,Inc': 'Univera Healthcare',                   # 2019-23 -> 2023-26
    'CenterLight Health System': 'CenterLight Healthcare Inc.',             # 2019-20 -> 2020-25
    'iCircle Care': 'iCircle Services of the Finger Lakes, Inc.',           # 2019-24 -> 2024-26
    'Fidelis Care New York': 'Fidelis Care',                                # 2019-24 / 2022-26 (overlap)
}


# =============================================================
# SMALL HELPERS
# =============================================================

def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _normkey(text: str) -> str:
    """Lower-case and drop everything except letters/digits (spacing/punctuation-proof key)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_normalized_map(mapping: dict, name: str) -> dict:
    """
    {normalised key -> canonical value}. Guards against silent mapping bugs:
      * two keys that normalise the same must point to the same value
      * a value must never be re-mapped to something else (no chained mappings)
    """
    norm = {}
    for key, value in mapping.items():
        nk = _normkey(key)
        if nk in norm and norm[nk] != value:
            raise ValueError(f"[{name}] conflicting mapping for '{key}': '{norm[nk]}' vs '{value}'")
        norm[nk] = value
    for value in set(norm.values()):
        tk = _normkey(value)
        if tk in norm and norm[tk] != value:
            raise ValueError(f"[{name}] chained mapping: '{value}' is itself re-mapped to '{norm[tk]}'")
    return norm


def map_value(value, norm_map: dict):
    """Apply a normalised map to one value (non-strings such as NaN pass through)."""
    if not isinstance(value, str):
        return value
    return norm_map.get(_normkey(value), value)


def build_autodedup_map(counts: pd.Series) -> dict:
    """
    Variants that differ only by spacing / punctuation / case are collapsed to the most
    frequent spelling (ties -> alphabetical, so the result is deterministic).
    """
    groups: dict[str, list] = {}
    for value, cnt in counts.items():
        key = _normkey(value)
        if key == "":                       # punctuation-only labels: never merge
            continue
        groups.setdefault(key, []).append((value, int(cnt)))
    remap = {}
    for variants in groups.values():
        if len(variants) > 1:
            canonical = sorted(variants, key=lambda x: (-x[1], x[0]))[0][0]
            for value, _ in variants:
                if value != canonical:
                    remap[value] = canonical
    return remap


def auto_dedup_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    values = df[col].dropna()
    values = values[values.map(lambda v: isinstance(v, str))]
    remap = build_autodedup_map(values.value_counts())
    if remap:
        df[col] = df[col].map(lambda v: remap.get(v, v) if isinstance(v, str) else v)
    print(f"[{col}] auto-dedup: {len(remap)} spelling variants collapsed.")
    return df


def _clean_text_value(value):
    """Strip + collapse repeated whitespace; blank -> NaN; non-strings become str; NaN stays NaN."""
    if pd.isnull(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    return np.nan if text == "" else text


# =============================================================
# STEP 1: LOAD
# =============================================================

def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            f"Put it at data/raw/{DEFAULT_RAW_PATH.name} inside the project, or pass --raw <path>."
        )
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Raw file is missing required columns: {missing}")

    # strict numeric conversion: fail loudly instead of silently creating NaNs
    for col in ["Year"] + PROCESS_DAY_COLS + REVENUE_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    print(f"Loaded: {path}")
    print(f"Shape: {df.shape}")
    return df


# =============================================================
# STEP 2: AUDIT (read-only)
# =============================================================

def basic_shape_info(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\n--- df.info() ---")
    df.info()
    print("\nDtype counts:\n", df.dtypes.value_counts())


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = (pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
              .sort_values("missing_pct", ascending=False))
    report = report[report["missing_count"] > 0]
    print(report)
    return report


def missing_vs_state_check(df: pd.DataFrame, missing_report: pd.DataFrame,
                           group_col: str = "State") -> None:
    for col in missing_report.index.tolist():
        tab = df.groupby(group_col)[col].apply(lambda x: x.isnull().mean() * 100).round(1)
        print(f"{col}:")
        print(tab.to_string())
        print("-" * 50)


def classify_features(df: pd.DataFrame) -> dict:
    identifier_cols = [ID_COL]
    date_like_cols = ["Year"]
    numeric_cols = [c for c in df.columns
                    if c not in identifier_cols + CATEGORICAL_COLS + TEXT_COLS + date_like_cols]
    buckets = {"identifier": identifier_cols, "categorical": CATEGORICAL_COLS,
               "text": TEXT_COLS, "numeric": numeric_cols, "date_like": date_like_cols}
    for name, cols in buckets.items():
        print(f"{name.upper()} ({len(cols)}): {cols}\n")
    return buckets


def feature_summary(df: pd.DataFrame, buckets: dict) -> None:
    print("=== CATEGORICAL FEATURES ===")
    for col in buckets["categorical"]:
        print(f"\n{col} - {df[col].nunique(dropna=True)} unique values")
        print(df[col].value_counts(dropna=False).head(10))
    print("\n\n=== NUMERIC FEATURES - describe() ===")
    print(df[buckets["numeric"]].describe().T)


def outlier_report(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    summary = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        summary.append({"column": col, "q1": q1, "q3": q3, "lower_bound": lower,
                        "upper_bound": upper, "n_outliers": n_outliers,
                        "pct_outliers": round(n_outliers / len(series) * 100, 2)})
    outlier_df = pd.DataFrame(summary)
    if not outlier_df.empty:
        outlier_df = outlier_df.sort_values("pct_outliers", ascending=False)
        print(outlier_df.to_string(index=False))
    return outlier_df


def duplicate_check(df: pd.DataFrame, id_col: str = ID_COL) -> None:
    """Exact numbers: extra rows caused by exact copies vs. same CaseID with different data."""
    extra_exact = int(df.duplicated().sum())
    extra_by_id = int(df.duplicated(subset=[id_col]).sum())
    conflicting = extra_by_id - extra_exact       # extra CaseID rows that are NOT exact copies
    print(f"Extra rows that are exact copies of another row : {extra_exact:,}")
    print(f"Extra rows sharing a {id_col} with different data: {conflicting:,}")


def run_audit(df: pd.DataFrame) -> None:
    _section("STEP 2: AUDIT (read-only)")
    basic_shape_info(df)
    report = missing_value_report(df)
    missing_vs_state_check(df, report)
    buckets = classify_features(df)
    feature_summary(df, buckets)
    outlier_report(df, buckets["numeric"])
    duplicate_check(df)


# =============================================================
# STEP 3: DEDUPLICATION (on raw values, before any standardisation)
# =============================================================

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(df):,} fully-identical duplicate rows. New shape: {df.shape}")
    return df


def resolve_duplicate_caseids(df: pd.DataFrame, id_col: str = ID_COL) -> pd.DataFrame:
    """
    Same CaseID but different data: keep the most complete row (fewest missing values;
    ties -> first occurrence) and flag those cases with Is_Amended_Group = 1.
    """
    df = df.copy()
    missing_count = df.isnull().sum(axis=1)
    df["Is_Amended_Group"] = 0

    dup_mask = df.duplicated(subset=[id_col], keep=False)
    n_groups = df.loc[dup_mask, id_col].nunique()
    if n_groups == 0:
        print("No duplicate CaseIDs left.")
        return df

    order = missing_count[dup_mask].sort_values(kind="stable").index
    keep_idx = df.loc[order].groupby(id_col).head(1).index
    drop_idx = df.index[dup_mask].difference(keep_idx)

    df.loc[keep_idx, "Is_Amended_Group"] = 1
    df = df.drop(index=drop_idx).reset_index(drop=True)
    print(f"Resolved {n_groups} duplicate CaseID groups: kept {len(keep_idx)}, dropped {len(drop_idx)}.")
    return df


# =============================================================
# STEP 4: CATEGORY STANDARDISATION
# =============================================================

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace / blank->NaN on categorical columns (mixed-type safe)."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].map(_clean_text_value)
    return df


def split_top_level(value: str) -> list:
    """Split on commas that are OUTSIDE parentheses, e.g. 'A (x, y), B' -> ['A (x, y)', 'B']."""
    parts, current, depth = [], [], 0
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return [p for p in parts if p]


def standardize_multivalue(df: pd.DataFrame, col: str, prefix: str, mapping: dict) -> pd.DataFrame:
    """
    Comma-separated multi-value categories:
      1. split into parts (commas inside brackets are NOT separators), map each part,
      2. collapse residual spelling variants (most frequent spelling wins),
      3. drop repeated parts, keep order.
    Creates {col}_Full, {prefix}_Primary, {prefix}_Count, {prefix}_HasMultiple.
    """
    df = df.copy()
    norm_map = build_normalized_map(mapping, col)

    def split_parts(value):
        if not isinstance(value, str):
            return None
        parts = [map_value(p, norm_map) for p in split_top_level(value)]
        parts = list(dict.fromkeys(parts))
        return parts or None

    parts = df[col].map(split_parts)
    flat = [p for lst in parts if isinstance(lst, list) for p in lst]
    dedup_map = build_autodedup_map(pd.Series(flat, dtype=object).value_counts())

    def finalize(lst):
        if not isinstance(lst, list):
            return None
        return list(dict.fromkeys(dedup_map.get(p, p) for p in lst))

    final = parts.map(finalize)
    df[col] = final.map(lambda l: ", ".join(l) if isinstance(l, list) else np.nan)
    df[f"{col}_Full"] = df[col]
    df[f"{prefix}_Primary"] = final.map(lambda l: l[0] if isinstance(l, list) else np.nan)
    df[f"{prefix}_Count"] = final.map(lambda l: len(l) if isinstance(l, list) else 0).astype(int)
    df[f"{prefix}_HasMultiple"] = (df[f"{prefix}_Count"] > 1).astype(int)

    n_multi = int(df[f"{prefix}_HasMultiple"].sum())
    print(f"[{col}] {n_multi:,} rows had multiple values; "
          f"{df[f'{prefix}_Primary'].nunique()} unique {prefix}_Primary values.")
    return df


def standardize_health_plan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norm_map = build_normalized_map(HEALTHPLAN_MAP, "HealthPlan")
    df["HealthPlan"] = df["HealthPlan"].map(lambda v: map_value(v, norm_map))
    df = auto_dedup_column(df, "HealthPlan")
    print("HealthPlan unique values now:", df["HealthPlan"].nunique())
    return df


def standardize_age_range(age_str) -> str:
    """
    Two numbers ('11 to 20', '50-59')  -> decade bucket of the midpoint ('10-19', '50-59').
    Open-ended ('65+', 'over 90')      -> '60+' (or 'N+' when N < 60).
    'Under 18' / '<18'                 -> '<N'.
    Missing / unparseable              -> 'Unknown'.

    Everything from 60 upwards is ONE bucket ('60+') because the two states use different
    schemes above 50 (CA: '51 to 64', '65+'; NY: '60-69', '70-79', '80-89', 'over 90').
    Known approximation: CA '51 to 64' is placed in '50-59' (its 60-64 part is lost).
    """
    if pd.isnull(age_str):
        return "Unknown"
    text = str(age_str).strip().lower()
    numbers = [int(n) for n in re.findall(r"\d+", text)]

    if len(numbers) >= 2:
        midpoint = (numbers[0] + numbers[1]) / 2
        if midpoint >= 60:
            return "60+"
        low = int(midpoint // 10) * 10
        return f"{low}-{low + 9}"

    if len(numbers) == 1:
        n = numbers[0]
        if any(h in text for h in ("+", "over", "older", "above", "plus", ">")):
            return "60+" if n >= 60 else f"{n}+"
        if any(h in text for h in ("under", "less than", "below", "<")):
            return f"<{n}"
    return "Unknown"


def standardize_gender(value) -> str:
    if pd.isnull(value):
        return "Unknown"
    key = str(value).strip().lower()
    return {"female": "Female", "f": "Female", "male": "Male", "m": "Male"}.get(key, "Other")


def standardize_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full standardisation. Must run BEFORE imputation. Returns (df, report_frames)."""
    df = normalize_text_columns(df)
    reports = {}

    # DenialType: 'Medical necessity' / 'Medical Necessity' -> one label
    df["DenialType"] = df["DenialType"].map(lambda v: v.title() if isinstance(v, str) else v)

    # Diagnosis / Treatment (multi-value)
    df = standardize_multivalue(df, "DiagnosisCategory", "Diagnosis", DIAGNOSIS_ABBREV_MAP)
    df = standardize_multivalue(df, "TreatmentCategory", "Treatment", TREATMENT_ABBREV_MAP)

    # Sub-categories: spelling variants only
    for col in ["DiagnosisSubCategory", "TreatmentSubCategory"]:
        df = auto_dedup_column(df, col)

    df = standardize_health_plan(df)

    # AgeRange (keep raw -> clean mapping for review)
    raw_age = df["AgeRange"].copy()
    df["AgeRange"] = raw_age.map(standardize_age_range)
    reports["age_range_mapping"] = (
        pd.DataFrame({"raw": raw_age, "clean": df["AgeRange"]})
        .value_counts(dropna=False).reset_index(name="rows")
        .sort_values(["clean", "raw"], na_position="last").reset_index(drop=True)
    )

    # Gender
    raw_gender = df["Gender"].copy()
    df["Gender"] = raw_gender.map(standardize_gender)
    print("\nGender mapping (raw -> clean):")
    print(pd.DataFrame({"raw": raw_gender, "clean": df["Gender"]})
          .value_counts(dropna=False).to_string())

    print("\nCategory standardisation done.")
    print("DenialType unique values now:", df["DenialType"].nunique())
    print("AgeRange values now:", sorted(df["AgeRange"].unique().tolist()))

    # value counts of the modelling categories, saved for manual review
    frames = []
    for col in ["Diagnosis_Primary", "Treatment_Primary", "HealthPlan",
                "DenialType", "AgeRange", "Gender"]:
        vc = df[col].value_counts(dropna=False).rename_axis("value").reset_index(name="rows")
        vc.insert(0, "column", col)
        frames.append(vc)
    reports["category_value_counts"] = pd.concat(frames, ignore_index=True)
    return df, reports


# =============================================================
# STEP 5-7: MISSING FLAGS -> OUTLIER CAP -> IMPUTATION
# =============================================================

def _train_mask(df: pd.DataFrame) -> pd.Series:
    mask = df["Year"] <= TRAIN_MAX_YEAR
    if not mask.any():
        print(f"WARNING: no rows with Year <= {TRAIN_MAX_YEAR}; using ALL rows for statistics.")
        mask = pd.Series(True, index=df.index)
    return mask


def revenue_lookup_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shows that the revenue columns are a category lookup, not case-level money.
    Must run after add_missing_flags() and BEFORE any imputation.
    """
    real = df[df["RealAvgMedicarePaid_Missing"] == 0]
    n_combo = real.groupby(REVENUE_COLS, dropna=False).ngroups
    print(f"{len(real):,} rows have real revenue values, but only {n_combo} distinct "
          f"combinations of {REVENUE_COLS}.")
    table = (df.groupby(["State", "Treatment_Primary"])
               .agg(rows=(ID_COL, "size"),
                    real_rows=("RealAvgMedicarePaid_Missing", lambda s: int((s == 0).sum())),
                    distinct_paid_values=("RealAvgMedicarePaid", "nunique"),
                    paid_min=("RealAvgMedicarePaid", "min"),
                    paid_max=("RealAvgMedicarePaid", "max"))
               .reset_index().sort_values("rows", ascending=False))
    none = table[table["real_rows"] == 0]
    print(f"{len(none)} of {len(table)} State x Treatment groups have NO real revenue value "
          f"({int(none['rows'].sum()):,} rows) -> left as NaN, use RealAvgMedicarePaid_Missing.")
    within = table[table["real_rows"] > 0]["distinct_paid_values"].max()
    print(f"Max distinct real Paid values inside any State x Treatment group: {within} "
          f"(1 = pure lookup: the number only encodes the category).")
    return table


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """{col}_Missing = 1 where the ORIGINAL value was missing. Must run before any fill/cap."""
    df = df.copy()
    for col in PROCESS_DAY_COLS + REVENUE_COLS:
        df[f"{col}_Missing"] = df[col].isnull().astype(int)
    return df


def cap_outliers(df: pd.DataFrame, cols_to_cap: list) -> pd.DataFrame:
    """
    Cap ONLY process-delay columns (revenue columns are left uncapped on purpose).
    Bounds use REAL values of the training years only. Real values above the bound are
    clipped; imputed rows are untouched (there are none yet -- this runs before imputation).
    """
    df = df.copy()
    train = _train_mask(df)
    for col in cols_to_cap:
        real = df[f"{col}_Missing"] == 0
        basis = df.loc[train & real, col]
        if basis.empty:
            print(f"{col}: no real training values, nothing capped.")
            continue
        q1, q3 = basis.quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        n_capped = int((df.loc[real, col] > upper).sum())
        df.loc[real, col] = df.loc[real, col].clip(upper=upper)
        print(f"{col}: {n_capped:,} real values capped at upper bound {upper:.2f} "
              f"(of {int(real.sum()):,} real values)")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    train = _train_mask(df)

    # (a) structural categorical: field simply not reported by that state
    for col in STRUCTURAL_CATEGORICAL:
        df[col] = df[col].fillna("Not_Reported")

    # (b) process-delay columns: median of REAL training values (flag column marks these rows)
    for col in PROCESS_DAY_COLS:
        real = df[f"{col}_Missing"] == 0
        median = df.loc[train & real, col].median()
        if pd.isnull(median):
            raise ValueError(f"{col}: no real training values to compute a median from.")
        n = int((~real).sum())
        df.loc[~real, col] = median
        print(f"{col}: {n:,} missing values filled with training median {median:.2f}")

    # (c) revenue lookup columns: NOT imputed by default (see IMPUTE_REVENUE_LOOKUP)
    for col in (REVENUE_COLS if IMPUTE_REVENUE_LOOKUP else []):
        real = df[f"{col}_Missing"] == 0
        basis = df.loc[train & real]
        global_median = basis[col].median()
        if pd.isnull(global_median):
            raise ValueError(f"{col}: no real training values to compute a median from.")
        group_median = basis.groupby(REVENUE_GROUP_KEYS)[col].median()
        keys = pd.MultiIndex.from_frame(df[REVENUE_GROUP_KEYS])
        fill = pd.Series(group_median.reindex(keys).to_numpy(), index=df.index)

        miss = ~real
        df.loc[miss, col] = fill[miss]
        leftover = df[col].isnull()
        df.loc[leftover, col] = global_median
        print(f"{col}: {int(miss.sum()):,} missing filled "
              f"({int(miss.sum() - leftover.sum()):,} by State x Treatment median, "
              f"{int(leftover.sum()):,} by global training median)")

    if not IMPUTE_REVENUE_LOOKUP:
        print(f"Revenue columns left as NaN ({', '.join(REVENUE_COLS)}); "
              f"see the *_Missing flags and revenue_lookup_table.csv.")

    # (d) free text
    df["Findings_or_Summary"] = df["Findings_or_Summary"].fillna("No summary provided")

    # AgeRange / Gender were already given an explicit 'Unknown' label (no mode imputation)
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    print("Imputation done. Remaining missing values:",
          "none" if remaining.empty else remaining.to_dict())
    return df


# =============================================================
# STEP 8: CMS COLUMNS
# =============================================================

def investigate_cms_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    CMS_Avg_* columns are provider-level aggregates, not case-level values. Columns with
    very few unique values carry no per-case signal (they are State proxies). For each
    column we also report how many unique values it has WITHIN a State (1 = pure proxy).
    """
    cms_cols = [c for c in df.columns if c.startswith(CMS_PREFIX)]
    print(f"\nTotal {CMS_PREFIX}* columns: {len(cms_cols)}")
    within_state = df.groupby("State")[cms_cols].nunique().max()

    report = pd.DataFrame({
        "column": cms_cols,
        "nunique": [df[c].nunique() for c in cms_cols],
        "max_nunique_within_state": [int(within_state[c]) for c in cms_cols],
    })
    report["pct_of_rows"] = (report["nunique"] / len(df) * 100).round(4)
    report = report.sort_values(["nunique", "column"]).reset_index(drop=True)

    low = report[report["nunique"] <= LOW_VARIANCE_MAX_UNIQUE]
    pure_proxy = low[low["max_nunique_within_state"] <= 1]
    print(f"{len(low)} columns have <= {LOW_VARIANCE_MAX_UNIQUE} unique values; "
          f"{len(pure_proxy)} of them are constant within each State (pure State proxies).")
    varying = low[low["max_nunique_within_state"] > 1]["column"].tolist()
    if varying:
        print("WARNING - low-variance but NOT constant within State (will still be dropped):", varying)
    return report


def drop_low_variance_cms_columns(df: pd.DataFrame, cms_report: pd.DataFrame,
                                  keep_one_reference: bool = True) -> pd.DataFrame:
    """Drop CMS columns with <= LOW_VARIANCE_MAX_UNIQUE unique values; optionally keep one reference column."""
    df = df.copy()
    to_drop = cms_report.loc[cms_report["nunique"] <= LOW_VARIANCE_MAX_UNIQUE, "column"].tolist()
    if keep_one_reference and CMS_REFERENCE_COL in to_drop:
        to_drop.remove(CMS_REFERENCE_COL)
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    print(f"Dropped {len(to_drop)} low-variance CMS columns.")
    if keep_one_reference:
        print(f"Kept 1 reference column: {CMS_REFERENCE_COL} "
              f"(still a State proxy -- do not use it as an independent signal).")
    print("New shape:", df.shape)
    return df


# =============================================================
# STEP 9: VALIDATE
# =============================================================

def validate_output(df: pd.DataFrame) -> None:
    problems = []
    allowed_nan = [] if IMPUTE_REVENUE_LOOKUP else REVENUE_COLS
    nan_cols = [c for c in df.columns[df.isnull().any()] if c not in allowed_nan]
    if nan_cols:
        problems.append(f"NaN values remain in: {nan_cols}")
    if df[ID_COL].duplicated().any():
        problems.append("Duplicate CaseIDs remain")
    bad = set(df[TARGET_COL].unique()) - {"Upheld", "Overturned"}
    if bad:
        problems.append(f"Unexpected {TARGET_COL} values: {sorted(bad)}")
    for col in PROCESS_DAY_COLS + REVENUE_COLS:
        if (df[col] < 0).any():
            problems.append(f"Negative values in {col}")
    for col in ["Diagnosis_Primary", "Treatment_Primary", "Is_Amended_Group"] + \
               [f"{c}_Missing" for c in PROCESS_DAY_COLS + REVENUE_COLS]:
        if col not in df.columns:
            problems.append(f"Expected column missing: {col}")
    if problems:
        raise ValueError("Validation failed:\n - " + "\n - ".join(problems))
    note = "" if IMPUTE_REVENUE_LOOKUP else f" (NaN only in the revenue lookup columns: {int(df['RealAvgMedicarePaid'].isnull().sum()):,} rows)"
    print(f"Validation passed: {len(df):,} rows, {df.shape[1]} columns, unique CaseIDs, no unexpected NaN{note}.")


# =============================================================
# PIPELINE
# =============================================================

def run_pipeline(raw_path=DEFAULT_RAW_PATH, interim_dir=DEFAULT_INTERIM_DIR,
                 reports_dir=DEFAULT_REPORTS_DIR, skip_audit: bool = False) -> pd.DataFrame:
    interim_dir, reports_dir = Path(interim_dir), Path(reports_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _section("STEP 1: LOAD")
    df = load_data(raw_path)

    if not skip_audit:
        run_audit(df)

    _section("STEP 3: DEDUPLICATION (raw values)")
    df = drop_exact_duplicates(df)
    df = resolve_duplicate_caseids(df)

    _section("STEP 4: STANDARDISE CATEGORIES / AGE / GENDER")
    df, reports = standardize_categories(df)

    _section(f"STEP 5: MISSING FLAGS  (statistics below use Year <= {TRAIN_MAX_YEAR} only)")
    df = add_missing_flags(df)
    revenue_table = revenue_lookup_report(df)

    _section("STEP 6: CAP OUTLIERS (process-delay only, real values only, revenue NOT capped)")
    df = cap_outliers(df, PROCESS_DAY_COLS)

    _section("STEP 7: IMPUTE MISSING VALUES")
    df = impute_missing(df)

    _section("STEP 8: CMS COLUMNS (investigate + drop low-variance State proxies)")
    cms_report = investigate_cms_columns(df)
    df = drop_low_variance_cms_columns(df, cms_report)

    _section("STEP 9: VALIDATE + SAVE")
    validate_output(df)

    out_path = interim_dir / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)
    cms_report.to_csv(reports_dir / "cms_column_investigation.csv", index=False)
    reports["age_range_mapping"].to_csv(reports_dir / "age_range_mapping.csv", index=False)
    reports["category_value_counts"].to_csv(reports_dir / "category_value_counts.csv", index=False)
    revenue_table.to_csv(reports_dir / "revenue_lookup_table.csv", index=False)

    print(f"\nCleaned data saved to: {out_path}")
    print(f"Review reports saved in: {reports_dir}")
    print("Final shape:", df.shape)
    return df


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master data-cleaning pipeline.")
    parser.add_argument("--raw", default=str(DEFAULT_RAW_PATH), help="path to the raw CSV")
    parser.add_argument("--interim-dir", default=str(DEFAULT_INTERIM_DIR), help="output folder for the cleaned CSV")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="output folder for review reports")
    parser.add_argument("--skip-audit", action="store_true", help="skip the long read-only audit printout")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.raw, args.interim_dir, args.reports_dir, args.skip_audit)