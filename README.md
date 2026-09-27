# Prior-Authorization Appeal Outcome Model

A machine-learning system that estimates the probability that a **denied** prior-authorization case will be
**overturned** on appeal, so that appeal teams can put their effort where it is most likely to pay off.
It is trained on real California and New York appeal decisions, tested on years the model never saw,
and served through a live dashboard.

---

## Overview

This project answers one question: *"This denial went to appeal. How likely is it to be overturned?"* It
is built on 103,680 real appeal decisions -- California (IMR) from 2001-2026 and New York (external
appeals) from 2019-2026 -- combined with CMS payment averages. A calibrated probability is produced for
each case, along with a per-State decision and a payment-proxy exposure figure, served through a live
dashboard that anyone on the appeals or billing team can use.

---

## Aim

- Estimate the probability that a denied case will be overturned on appeal
- Give appeal and billing teams a clear "likely overturned" / "likely upheld" decision per case
- Show how similar past cases went, by State, treatment, and diagnosis
- Express the money at stake through a clearly labeled payment-proxy estimate
- Present everything through a simple, live, interactive dashboard

---

## Problem Statement

When a prior-authorization request is denied, the provider can appeal -- but every appeal costs staff time,
and teams cannot fight every denial with the same effort. Deciding which denials are worth appealing is
usually left to experience and gut feeling, so effort is often spent on weak cases while winnable ones are
dropped.

Three things make this decision hard:

- **Outcomes differ by State and change over time.** In this data the overturn rate rose from about 47% in
  2001-2023 to about 56% in 2025-2026, and California and New York behave very differently.
- **A raw model score is not a real probability.** Without calibration, a score of "0.6" does not reliably
  mean a 60% chance, so it cannot be trusted for planning.
- **The money at stake is not in the public data.** Appeal records do not include claim amounts, so any
  money view has to be a clearly labeled estimate.

The goal is to give appeal and billing teams a trustworthy, State-aware probability for each denied case,
with historical context and a clearly labeled money estimate, so effort goes where it is most likely to pay off.

---

## Key Features

- Case-level probability that a denial is overturned, calibrated per State
- Live **Case Checker** -- score any real case and see the decision, payment proxy, and historical context
- **Trends and Drift** view -- how overturn rates are changing year over year, per State
- Payment proxy with expected recoverable vs. expected lost amounts
- Simple, one-command dashboard -- every number shown comes from the real model, not a static demo

---

## Architecture

### End-to-end system

![End-to-end system architecture](docs/assets/01_system_architecture.gif)

### What happens when a case is scored

![Case-scoring sequence](docs/assets/02_case_scoring_sequence.gif)

---

## Dashboard / Results

**Case Checker** -- describe a denied case and get the real model's calibrated chance of it being
overturned, along with a payment-proxy estimate and how similar cases went historically.

<img src="docs/assets/screenshots/01_case_checker.png" width="800">

**Trends and Drift** -- shows how the overturn rate has changed year over year, differently in each
State, which is why the model is calibrated and evaluated separately per State.

<img src="docs/assets/screenshots/02_trends.png" width="800">

---

## Benefits

- **Better prioritization** -- teams can put their limited appeal effort where it is most likely to pay off
- **State-aware accuracy** -- California and New York are evaluated and calibrated separately, since their
  overturn patterns differ
- **Transparent probabilities** -- a "60% chance" shown on screen really does mean about 60%, not an
  inflated or misleading number
- **Historical context** -- every case comes with how similar past cases (same State, treatment, or
  diagnosis) actually turned out
- **Clear money view** -- a labeled payment-proxy estimate, never presented as an actual claim amount
- **One live dashboard** -- no spreadsheets or manual lookups; billing and appeals staff get an answer in
  seconds

---

## Conclusion

The model was trained on decisions from 2001-2023, calibrated on 2024, and then tested on 22,649 decisions
from 2025-2026 that it had never seen. On that unseen period it separates likely-overturned from
likely-upheld cases with an AUC of about 0.71 overall (about 0.73 in California and 0.69 in New York), and
per-State calibration brings the displayed probabilities closer to the real outcome rates.

This makes it a practical tool for ranking and prioritizing appeals: it helps teams decide where to spend
their effort first, while final decisions stay with the people who review each case. Because overturn rates
keep shifting over time, the Trends and Drift view and regular retraining on new decisions are part of
keeping the model reliable.