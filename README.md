# Prior-Authorization Appeal Outcome Model

A machine-learning system that estimates the probability that a **denied** prior-authorization case will be
**overturned** on appeal, so that appeal teams can put their effort where it is most likely to pay off.
It is trained on real California and New York appeal decisions, tested on later years the model never saw,
and served through a small API and a React dashboard.

> **What it is not:** it does not know real claim amounts, and it is not a medical or coverage decision tool.
> The money view is a **payment proxy** (an average Medicare payment per service, looked up by treatment label).

---

## 1. Overview

| | |
|---|---|
| **Question answered** | "This denial went to appeal. How likely is it to be overturned?" |
| **Output** | A calibrated probability per case, a per-State decision ("likely overturned" / "likely upheld"), and a payment-proxy exposure |
| **Users** | Billing / appeals teams deciding which denied cases to fight first |
| **Data** | 103,680 appeal decisions: California (IMR) 2001-2026 and New York (external appeals) 2019-2026, plus CMS payment averages |
| **Model** | XGBoost, calibrated per State, evaluated with time-aware validation |
| **Interface** | FastAPI backend + React dashboard (one command to run) |

---

## 2. Problem statement

When an insurer denies a service (often because prior authorization was missing or not accepted), the provider can
appeal. An independent reviewer then either **upholds** the denial (the insurer was right) or **overturns** it (the
payment must be made). Appeals cost time and money, and a team can only fight a limited number of them.

**Goal:** rank denied cases by their chance of being overturned, and express the money at stake in a way that is honest
about what the data can support.

**Success means:**
1. The ranking is better than chance on years the model has never seen (measured per State, with confidence intervals).
2. The probabilities are calibrated (a "60%" case really is overturned about 60% of the time).
3. No information from after the decision leaks into the model.
4. Every number shown to a user comes from the real model, and every limitation is stated.

---



---

## 4. What we did

### Pipeline

| Step | Script | What it does | Why |
|---|---|---|---|
| 1 | `data_cleaning.py` | Removes duplicates, unifies category labels, handles missing values, caps outliers on real values only | Clean, consistent input; statistics learned from training years only |
| 2 | `check_leakage_columns.py` | Tests every candidate feature alone, out of time; checks split drift | Detect leakage and drift before modelling |
| 3 | `eda_core.py` | 20 charts and effect-size tests, split by State | Understand the data; pooled charts would hide the State differences |
| 4 | `feature_engineering.py` | Time-based split, one-hot and out-of-fold target encoding, scaling, feature selection; saves the fitted pipeline | Matrices for modelling; the same pipeline is reused for new cases |
| 5 | `tune_model.py` | Randomised search plus local refinement of XGBoost parameters | Better model settings |
| 6 | `train_model.py` | Trains 3 models, picks the best, fits per-State calibrators and thresholds on the validation year, evaluates on test once | Honest, calibrated evaluation |
| 7 | `revenue_proxy.py` | Expected recoverable / lost = probability x proxy amount; compares with what happened | Transparent money view |
| 8 | `api.py` + dashboard | Serves the real model to a web page | Use and demonstration |

### Key design decisions
- **Time-based split:** train up to 2023, validation 2024, test 2025-2026. The test years are loaded once, at the very end.
- **No leakage:** fields that are only known after the decision (`DaysToReview`, `DaysToAdopt`, findings text) are excluded.
  Every statistic (medians, outlier bounds, encodings, scaler) comes from the training years only. Verified: flipping every
  test label leaves the model, thresholds and calibrators unchanged.
- **Out-of-fold target encoding** for high-cardinality fields (diagnosis, treatment, health plan), so no row sees its own label.
- **Per-State calibration and thresholds:** the overturn rate rose after the training years, so raw probabilities are too low;
  a per-State Platt calibrator and a per-State threshold are fitted on the validation year.
- **Payment proxy instead of a revenue model:** see below.

### Problems found and corrected (audit trail)
| Found | Correction |
|---|---|
| A "revenue" regression scored R2 = 0.999 | It was learning a category lookup, not money. The regression was removed and replaced by the payment proxy |
| Outlier caps were computed after imputation and flattened real values | Caps now use real values only, from training years |
| Preprocessing statistics used all rows, including test years | Learned from training years only |
| Cross-validation was shuffled although the data is time-ordered | Time-aware (forward-chaining) validation |
| Decision threshold was inspected on the test set | Chosen on the validation year, per State |
| The dashboard was a static demo | Rebuilt as a React app that calls the real model |

---



---

## 6. Key features

- Case-level probability that a denial is overturned, calibrated per State
- Time-aware validation, leakage checks, and out-of-fold encodings
- Confidence intervals for the AUC (bootstrap) and calibration reported per State
- Payment proxy with expected vs actual comparison, and an explicit "no proxy available" bucket (nothing is imputed)
- Live **Case checker**: score any case with the real model, per-State fields, threshold and proxy
- Dashboard tabs: Overview, Case checker, Trends and drift, Payment proxy, Model and evidence (with a plain-language glossary)
- One-command server; the dashboard has no numbers that do not come from the model
- Automatic checks: 23 end-to-end dashboard checks, and API output equal to offline predictions (difference about 1e-16)

---

## 7. Architecture

### 7.1 End-to-end system

```mermaid
flowchart LR
    subgraph SRC["1. Data sources"]
        S1["California IMR<br/>appeal decisions"]
        S2["New York DFS<br/>external appeals"]
        S3["CMS Medicare<br/>payment averages"]
    end

    subgraph PREP["2. Data preparation"]
        P1["data_cleaning.py<br/>dedupe, unify labels,<br/>train-years-only statistics"]
        P2["check_leakage_columns.py<br/>leakage and drift checks"]
        P3["eda_core.py<br/>20 charts, effect sizes"]
        P4["feature_engineering.py<br/>time split, encodings,<br/>fitted pipeline"]
    end

    subgraph MODEL["3. Modelling"]
        M1["tune_model.py<br/>time-aware search"]
        M2["train_model.py<br/>3 models, calibration,<br/>thresholds, test once"]
        M3[("final_model.joblib<br/>predictions_val / test")]
    end

    subgraph MONEY["4. Payment proxy"]
        R1["revenue_proxy.py<br/>probability x proxy amount"]
    end

    subgraph SERVE["5. Serving"]
        A1["api.py - FastAPI<br/>predict, summary, trends, exposure"]
        UI["React dashboard<br/>Overview, Case checker, Trends,<br/>Payment proxy, Model"]
    end

    U(("Appeals and<br/>billing team"))

    S1 --> P1
    S2 --> P1
    S3 --> P1
    P1 --> P2
    P1 --> P3
    P1 --> P4
    P4 --> M1
    P4 --> M2
    M1 -. best parameters .-> M2
    M2 --> M3
    M3 --> R1
    M3 --> A1
    R1 --> A1
    A1 <--> UI
    UI --- U

    classDef src fill:#e8f1fa,stroke:#2E86AB,color:#0b3954
    classDef prep fill:#eaf7f5,stroke:#2A9D8F,color:#0b3d38
    classDef model fill:#fff4e5,stroke:#F4A261,color:#5a3200
    classDef serve fill:#fdeeec,stroke:#E63946,color:#5c0f16
    classDef user fill:#f0f0f0,stroke:#555555,color:#222222
    class S1,S2,S3 src
    class P1,P2,P3,P4 prep
    class M1,M2,M3,R1 model
    class A1,UI serve
    class U user
```

Solid arrows show data and files flowing between stages. The dotted arrow is the tuned-parameter file that
`train_model.py` reads (it never imports `tune_model.py`). The test years are used only by the last step of `train_model.py`.

### 7.2 What happens when a case is scored

```mermaid
sequenceDiagram
    actor User as Appeals team
    participant UI as React dashboard
    participant API as FastAPI api.py
    participant FE as Fitted feature pipeline
    participant M as XGBoost model
    participant C as Per-State calibrator
    User->>UI: Enter a denied case (State, diagnosis, treatment, ...)
    UI->>API: POST /api/predict
    API->>FE: Encode the case (same pipeline as training)
    FE->>M: 37 features
    M->>C: Raw probability
    C->>API: Calibrated probability
    API->>API: Compare with the State threshold, look up the payment proxy
    API->>UI: Probability, decision, proxy, context, warnings
    UI->>User: Show the result
```

The dashboard contains no numbers of its own: every value on screen comes from this request path or from the saved
evaluation files.

---

## 10. Honest limitations

1. **Moderate discrimination.** AUC is about 0.69 (NY) and 0.68 (CA). The model ranks cases; it does not decide them.
2. **Drift.** The overturn rate keeps rising. After calibration CA is still about 7 points too low (64.9% vs 71.9%).
   Calibrators (and the model, if needed) must be re-fitted whenever a new labelled year is available.
3. **CA is small in the test years** (3,268 cases), so its intervals are wide.
4. **Two populations.** CA and NY differ in label schemes, years covered and overturn rate. Many treatments and plans exist in one State only, so State is partly mixed into the features. Report results per State.
5. **No real dollar amounts.** The payment figure is an average Medicare payment per service, looked up by treatment label.
   Read totals as an exposure index, do not compare dollars between States, and note that 9-17% of cases have no proxy.
6. **Age buckets are approximate.** CA's "51 to 64" is placed in "50-59"; everything from 60 upwards is one bucket.
7. **Fairness.** `Gender` and `AgeRange` are inputs. Their association with the outcome is small (Cramer's V 0.02 and 0.09), but a fairness review is required before operational use.
8. **Public, aggregated data only.** There is no clinical documentation, so medical necessity itself cannot be modelled.
9. **No forecast.** The data has no month or date column (year only), so a volume forecast would not be reliable. This is future work.

---

## 11. Conclusion

The project answers a concrete, useful question with a model that has been tested honestly: on later years it never saw,
it ranks denied cases clearly better than chance (AUC about 0.69), it is calibrated per State, and its payment proxy tracks
what actually happened to within about 5%. Equally important, the work found and removed two misleading results (a
"revenue" model that only memorised a lookup table, and optimistic validation), and it states what the data cannot support.

The model is a **prioritisation aid for appeal teams**, not an automatic decision maker. Its next useful steps are:
re-fitting the calibrators each year, a fairness review, adding real claim amounts (which would turn the proxy into real
revenue), and a State-level monitoring dashboard for drift.

