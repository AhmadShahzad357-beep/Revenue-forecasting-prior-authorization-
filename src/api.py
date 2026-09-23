"""
src/api.py
==========
Serves the TRAINED model and the React dashboard from one small web server.

Run (works from any folder), AFTER train_model.py and revenue_proxy.py:
    pip install fastapi uvicorn
    python src/api.py                 # then open  http://127.0.0.1:8000
    python src/api.py --port 8080

Endpoints (all JSON)
  GET  /api/health     is everything loaded?
  GET  /api/summary    model, metrics with confidence intervals, drift, feature importance, limitations
  GET  /api/trends     overturn rate and case volume per Year and State (from the cleaned data)
  GET  /api/options    the values a case may take, per State
  POST /api/predict    score ONE case with the real model + calibrator + threshold + payment proxy
  GET  /api/exposure   expected vs actual payment-proxy totals (from revenue_proxy.py)

The dashboard page itself comes from dashboard/dist/index.html if that file exists, otherwise from the copy
compressed inside dashboard_page.py (so no extra file has to be copied).

Nothing here is a mock-up: /api/predict runs feature_engineering.transform() (the same fitted pipeline used
in training), the saved model, the per-State calibrator and the per-State threshold.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import feature_engineering as fe
import revenue_proxy as rp
import train_model as tm

try:
    import dashboard_page                 # the built dashboard, compressed in one Python file
except ImportError:                       # pragma: no cover
    dashboard_page = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned_dataset_final.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_OUT = PROJECT_ROOT / "outputs" / "models"
REPORTS = PROJECT_ROOT / "outputs" / "reports"
DIST_DIR = PROJECT_ROOT / "dashboard" / "dist"

STATES = ["CA", "NY"]
MIN_OPTION_ROWS = 20          # a value must occur at least this often in a State to be offered
NOT_REPORTED = "Not_Reported"
RECENT_FROM_YEAR = 2023       # "context" rates use the recent years (the target drifts)

# which fields exist in which State (the other State's rows carry 'Not_Reported')
STATE_FIELDS = {"CA": {"review_speed"}, "NY": {"health_plan", "coverage_type"}}
FIELD_COLUMN = {"diagnosis": "Diagnosis_Primary", "treatment": "Treatment_Primary", "denial_type": "DenialType",
                "age_range": "AgeRange", "gender": "Gender", "health_plan": "HealthPlan",
                "coverage_type": "CoverageType", "review_speed": "ReviewSpeed_IMRType"}

LIMITATIONS = [
    "Moderate discrimination: AUC is about 0.69 (NY) and 0.66 (CA). The model ranks cases better than chance, it does not decide them.",
    "The overturn rate keeps rising, most in CA. Calibrators are fitted on the validation year, so recent CA probabilities can still be too low.",
    "CA and NY use different label schemes; many treatments, plans and coverage types exist in only one State.",
    "There are no case-level dollar amounts. The payment figure is a per-service Medicare average looked up by treatment label (a proxy).",
    "Only public, aggregated CMS data is used; no clinical documentation, so medical necessity itself cannot be modelled.",
]


# =============================================================
# LOADING
# =============================================================

def _clean(obj):
    """Make anything JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _records(df: pd.DataFrame) -> list:
    return _clean(df.astype(object).where(pd.notna(df), None).to_dict("records"))


def _read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


class Store:
    """Everything loaded once at start-up; `missing` lists what is not there yet."""

    def __init__(self) -> None:
        self.missing: list = []
        self.bundle = self.art = self.data = self.lookup = None
        self.options: dict = {}
        self.load()

    def load(self) -> None:
        self.missing = []
        for label, path in [("models/final_model.joblib", MODELS_DIR / "final_model.joblib"),
                            ("models/feature_pipeline.joblib", MODELS_DIR / "feature_pipeline.joblib"),
                            ("data/interim/cleaned_dataset_final.csv", DATA_PATH)]:
            if not path.exists():
                self.missing.append(label)
        if self.missing:
            return
        self.bundle = joblib.load(MODELS_DIR / "final_model.joblib")
        self.art = joblib.load(MODELS_DIR / "feature_pipeline.joblib")
        self.data = pd.read_csv(DATA_PATH, low_memory=False,
                                usecols=["CaseID", "Year", "State", "Outcome", "Diagnosis_Primary", "Treatment_Primary",
                                         "HealthPlan", "CoverageType", "DenialType", "AgeRange", "Gender",
                                         "ReviewSpeed_IMRType"])
        self.data["y"] = (self.data["Outcome"] == "Overturned").astype(int)
        try:
            self.lookup = rp.load_lookup(REPORTS)
        except FileNotFoundError:
            self.lookup = {}
            self.missing.append("outputs/reports/revenue_lookup_table.csv (payment proxy disabled)")
        self.options = self._build_options()

    def _build_options(self) -> dict:
        out = {}
        for st in STATES:
            d = self.data[self.data["State"] == st]
            block = {}
            for field, col in FIELD_COLUMN.items():
                vc = d[col].value_counts()
                vc = vc[vc >= MIN_OPTION_ROWS]
                if field in ("health_plan", "coverage_type", "review_speed"):
                    if field in STATE_FIELDS[st]:
                        vc = vc.drop(labels=[NOT_REPORTED], errors="ignore")
                    else:
                        vc = pd.Series({NOT_REPORTED: int((d[col] == NOT_REPORTED).sum())})
                if field in ("age_range", "gender"):
                    vc = vc.drop(labels=["Unknown"], errors="ignore")
                block[field] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
            block["applicable_fields"] = sorted(STATE_FIELDS[st])
            out[st] = block
        return out

    def ready(self) -> bool:
        return self.bundle is not None and self.art is not None and self.data is not None


store = Store()
app = FastAPI(title="Prior-Authorization Appeal Outcome API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def _require_ready() -> None:
    if not store.ready():
        raise HTTPException(status_code=503, detail="Model files are missing: " + ", ".join(store.missing)
                            + ". Run data_cleaning.py, feature_engineering.py and train_model.py first.")


# =============================================================
# ROUTES
# =============================================================

@app.get("/api/health")
def health():
    return {"status": "ok" if store.ready() else "not_ready", "ready": store.ready(), "missing": store.missing,
            "model": store.bundle["model_name"] if store.ready() else None}


@app.get("/api/summary")
def summary():
    _require_ready()
    model = store.bundle["model"]
    importance = []
    if hasattr(model, "feature_importances_"):
        s = pd.Series(model.feature_importances_, index=store.bundle["features"]).sort_values(ascending=False)
        importance = [{"feature": k, "importance": float(v)} for k, v in s.head(15).items()]
    d = store.data
    return _clean({
        "model": {"name": store.bundle["model_name"], "n_features": len(store.bundle["features"]),
                  "thresholds": store.bundle["thresholds"], "train_max_year": store.art["train_max_year"],
                  "params": store.bundle.get("xgb_params")},
        "data": {"rows": int(len(d)), "years": [int(d["Year"].min()), int(d["Year"].max())],
                 "states": {st: {"rows": int((d["State"] == st).sum()),
                                 "years": [int(d[d["State"] == st]["Year"].min()), int(d[d["State"] == st]["Year"].max())],
                                 "overturn_rate": float(d[d["State"] == st]["y"].mean())} for st in STATES}},
        "test_metrics": _records(_read_csv(MODELS_OUT / "test_metrics.csv")) if (MODELS_OUT / "test_metrics.csv").exists() else [],
        "test_threshold_metrics": _records(_read_csv(MODELS_OUT / "test_threshold_metrics.csv")) if (MODELS_OUT / "test_threshold_metrics.csv").exists() else [],
        "cv": _records(_read_csv(MODELS_OUT / "cv_scores.csv")) if (MODELS_OUT / "cv_scores.csv").exists() else [],
        "comparison": _records(_read_csv(MODELS_OUT / "train_val_comparison.csv")) if (MODELS_OUT / "train_val_comparison.csv").exists() else [],
        "split_summary": _records(_read_csv(REPORTS / "split_summary.csv")) if (REPORTS / "split_summary.csv").exists() else [],
        "single_feature_screen": _records(_read_csv(REPORTS / "single_feature_screen.csv")) if (REPORTS / "single_feature_screen.csv").exists() else [],
        "feature_importance": importance,
        "limitations": LIMITATIONS,
    })


@app.get("/api/trends")
def trends():
    _require_ready()
    g = (store.data.groupby(["Year", "State"])["y"].agg(cases="size", overturn_rate="mean").reset_index())
    return _clean({"rows": _records(g), "train_max_year": store.art["train_max_year"]})


@app.get("/api/options")
def options():
    _require_ready()
    return {"states": STATES, "by_state": store.options, "min_rows": MIN_OPTION_ROWS}


class CaseIn(BaseModel):
    state: Literal["CA", "NY"]
    diagnosis: str
    treatment: str
    denial_type: str
    age_range: str
    gender: str
    health_plan: str = NOT_REPORTED
    coverage_type: str = NOT_REPORTED
    review_speed: str = NOT_REPORTED
    n_diagnoses: int = Field(1, ge=1, le=9)
    multiple_treatments: bool = False


def _context(state: str, treatment: str, diagnosis: str) -> dict:
    d = store.data
    recent = d[(d["State"] == state) & (d["Year"] >= RECENT_FROM_YEAR)]

    def rate(sub):
        return {"rate": float(sub["y"].mean()), "n": int(len(sub))} if len(sub) else {"rate": None, "n": 0}

    return {"since_year": RECENT_FROM_YEAR, "state": rate(recent),
            "treatment": rate(recent[recent["Treatment_Primary"] == treatment]),
            "diagnosis": rate(recent[recent["Diagnosis_Primary"] == diagnosis])}


@app.post("/api/predict")
def predict(case: CaseIn):
    _require_ready()
    st = case.state
    warnings: list = []

    values = {f: getattr(case, f) for f in FIELD_COLUMN}
    for f in ("health_plan", "coverage_type", "review_speed"):
        if f not in STATE_FIELDS[st]:
            if values[f] != NOT_REPORTED:
                warnings.append(f"'{f}' is not recorded in {st}; it was ignored.")
            values[f] = NOT_REPORTED
    for f, v in values.items():
        allowed = {o["value"] for o in store.options[st][f]}
        if v not in allowed:
            raise HTTPException(status_code=422, detail=f"'{v}' is not an available {f.replace('_', ' ')} value for {st}.")

    row = pd.DataFrame([{
        "State": st, "Gender": values["gender"], "DenialType": values["denial_type"],
        "CoverageType": values["coverage_type"], "AgeRange": values["age_range"],
        "ReviewSpeed_IMRType": values["review_speed"], "Diagnosis_Primary": values["diagnosis"],
        "Treatment_Primary": values["treatment"], "HealthPlan": values["health_plan"],
        "Diagnosis_Count": case.n_diagnoses, "Diagnosis_HasMultiple": int(case.n_diagnoses > 1),
        "Treatment_HasMultiple": int(case.multiple_treatments)}])

    X = fe.transform(row, store.art)                       # the SAME fitted pipeline used in training
    raw = float(store.bundle["model"].predict_proba(X[store.bundle["features"]])[0, 1])
    cal = float(tm.apply_calibration(np.array([raw]), np.array([st]), store.bundle["calibrators"])[0])
    thr = float(store.bundle["thresholds"][st])

    amount = store.lookup.get((st, values["treatment"])) if store.lookup else None
    proxy = {"available": amount is not None, "amount": amount,
             "expected_recoverable": cal * amount if amount is not None else None,
             "expected_lost": (1 - cal) * amount if amount is not None else None,
             "note": ("Per-service Medicare average for this treatment label (a proxy, not the claim amount)."
                      if amount is not None else "No payment proxy exists for this treatment label in this State.")}
    warnings.append("The overturn rate has been rising; recent probabilities may still be a little low"
                    + (" (especially in CA)." if st == "CA" else "."))
    return _clean({
        "state": st, "probability": cal, "raw_probability": raw, "threshold": thr,
        "decision": "likely overturned" if cal >= thr else "likely upheld",
        "margin": cal - thr, "proxy": proxy, "context": _context(st, values["treatment"], values["diagnosis"]),
        "warnings": warnings,
        "explain": "Probability that the denial is OVERTURNED on appeal, calibrated per State on the 2024 validation year.",
    })


@app.get("/api/exposure")
def exposure():
    _require_ready()
    try:
        tables = rp.build_tables(MODELS_OUT, REPORTS)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _clean({
        "by_year_state": _records(tables["by_year_state"]),
        "by_treatment": _records(tables["by_treatment"].head(20)),
        "overall": _records(tables["overall"]),
        "definitions": {
            "expected_recoverable": "sum of P(Overturned) x proxy amount (cases with a proxy only)",
            "expected_lost": "sum of P(Upheld) x proxy amount",
            "actual_recoverable": "sum of proxy amounts of the cases that really were overturned",
            "proxy": "average Medicare payment per service for the treatment label; NOT the size of a claim",
        },
    })


# =============================================================
# STATIC DASHBOARD
# =============================================================

if (DIST_DIR / "index.html").exists():            # an empty dist folder must NOT count as a dashboard
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="dashboard")
elif dashboard_page is not None:
    @app.get("/", response_class=HTMLResponse)
    def embedded_dashboard():
        return dashboard_page.get_html()
else:
    @app.get("/", response_class=HTMLResponse)
    def no_dashboard():
        return ("<h3>API is running.</h3><p>The dashboard page was not found. Copy dashboard_page.py into src, "
                "or open <a href='/docs'>/docs</a> to try the API.</p>")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Model API + dashboard server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    print(f"Model loaded: {store.ready()}  {('missing: ' + ', '.join(store.missing)) if store.missing else ''}")
    print(f"Open http://{a.host}:{a.port}   (API docs: http://{a.host}:{a.port}/docs)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()