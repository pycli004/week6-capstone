# ChurnGuard — Monitoring Notes

## Purpose

This document outlines what to monitor in production, why, and how often. It describes conceptual triggers, not automated enforcement.

## Logged Data

Every `/predict` request appends to `logs/predictions.csv`:

| Field | Description |
|---|---|
| `timestamp` | ISO-8601 UTC time |
| `probability` | Model output (0-1) |
| `prediction` | "Churn" or "No Churn" |
| `top_driver_1/2/3` | Feature names of top 3 SHAP contributors |

**Note:** Raw customer input is not currently logged for privacy. If drift monitoring requires it, enable logging of inputs to a separate secured file.

## Feature Distributions to Monitor

### 1. `tenure`
- **What to check:** Distribution should remain roughly bimodal (new customers <12 months, loyal customers >48 months).
- **Drift signal:** Mean tenure drops below 20 months (new-customer flood) or rises above 40 months (aging base).
- **Frequency:** Weekly.

### 2. `MonthlyCharges`
- **What to check:** Median and 75th percentile stable within ±10% of training values.
- **Drift signal:** Median shifts >15% or a new mode appears (pricing changes).
- **Frequency:** Weekly.

### 3. `Contract`
- **What to check:** Month-to-month share should stay 45-60%.
- **Drift signal:** Month-to-month share exceeds 70% or falls below 30% (portfolio shift).
- **Frequency:** Monthly.

### 4. `TotalCharges`
- **What to check:** Ratio of `TotalCharges / tenure` should track `MonthlyCharges` (sanity check).
- **Drift signal:** Systematic divergence suggests billing data quality issue.
- **Frequency:** Weekly.

### 5. `PaymentMethod`
- **What to check:** "Electronic check" share should stay 30-45%.
- **Drift signal:** Sudden jump suggests acquisition-channel change.
- **Frequency:** Monthly.

## Prediction Distribution Monitoring

### Churn Rate
- **Expected:** 20-35% of predictions are "Churn" (matches training class balance).
- **Drift signal:**
  - Churn rate > 40% for 7 consecutive days (model over-firing).
  - Churn rate < 10% for 7 consecutive days (model under-firing).
- **Action:** Investigate feature distribution shift before retraining.

### Probability Histogram
- **What to check:** No single probability bucket (>0.1 width) exceeds 50% of predictions.
- **Drift signal:** Model becomes overconfident (most predictions near 0.05 or 0.95).
- **Frequency:** Weekly.

## Top Driver Stability

- **What to check:** Top 3 features should match training-time top 3 (`ContractRisk`, `tenure`, `MonthlyCharges`).
- **Drift signal:** A new feature enters top 3 and persists for 2+ weeks.
- **Interpretation:** Business pattern changed; model may need retraining.

## Ground Truth Collection

ChurnGuard does not currently observe actual churn labels in production. To close the loop:

1. **Quarterly:** Join `logs/predictions.csv` against customer churn records (available 30-90 days after prediction).
2. **Compute:** Precision, recall, F1 on the ground-truth-joined set.
3. **Compare:** Against the reported test metrics (recall 0.8155, F1 0.6289).
4. **Drift signal:** Live recall drops below 0.70 → retrain.

## Retraining Trigger

Retrain the model when **any two** of the following are true:

1. Ground-truth recall < 0.70 (from quarterly join).
2. `ContractRisk` or `tenure` distribution shifts by more than 20% vs training baseline.
3. Churn rate in predictions exceeds 40% or falls below 10% for 14 days.
4. A new top-3 driver appears and persists for 3+ weeks.
5. New customer segment appears (e.g., new country, new pricing tier) not represented in training.

**Retraining procedure:**

1. Pull fresh labeled data.
2. Re-run `src/features.py` (feature engineering unchanged).
3. Re-run `src/train.py` (re-tune hyperparameters on new data).
4. Evaluate on a fresh hold-out — must beat current model on recall and F1.
5. If beaten, promote new model artifact; otherwise keep current.
6. Log retrain event with timestamp and metrics delta.

## Alerting (Conceptual)

If this system were in production, alerts would fire when:

- **High priority (immediate):** Backend `/health` fails, or `/predict` error rate > 5%.
- **Medium priority (daily):** Churn rate or probability distribution drifts outside expected range.
- **Low priority (weekly):** Feature distribution monitors deviate >20% from baseline.

Alerts would route to the ML engineer on-call and the retention team lead.

## What is Not Monitored (Out of Scope)

- Individual customer prediction accuracy (no per-customer label feedback loop yet).
- Latency drift (backend responds sub-2s in testing; production latency would need separate monitoring).
- Request volume anomalies (DDoS or misuse detection — handled at infrastructure level).

## Maintenance Calendar

| Frequency | Task |
|---|---|
| Weekly | Review probability histogram, top drivers, feature distribution snapshots |
| Monthly | Review contract/payment-method mix changes |
| Quarterly | Join predictions vs ground truth, compute live recall; decide on retraining |
| Annually | Full model audit against current customer population |