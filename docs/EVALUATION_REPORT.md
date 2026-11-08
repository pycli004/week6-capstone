# ChurnGuard — Evaluation Report

## Summary

The tuned XGBoost model achieves a **test-set recall of 0.8155** on churners (catching 81.5% of customers who will churn) with an **ROC-AUC of 0.8463** and **PR-AUC of 0.6531**. This is a large improvement over the majority-class baseline, which catches zero churners despite 73.5% accuracy.

## Dataset

- **Source:** Telco Customer Churn (Kaggle)
- **Total:** 7,043 customers, 21 raw columns
- **Target:** `Churn` (binary) — 5,174 No (73.5%), 1,869 Yes (26.5%)
- **Split:** Stratified 80/20 → 5,634 train / 1,409 test

## Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (majority) | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2650 |
| Logistic Regression | 0.7367 | 0.5026 | 0.7727 | 0.6091 | 0.8461 | 0.6645 |
| Random Forest | 0.7587 | 0.5324 | 0.7460 | 0.6214 | 0.8386 | 0.6478 |
| XGBoost (original) | 0.7601 | 0.5340 | 0.7567 | 0.6261 | 0.8355 | 0.6448 |
| **XGBoost (tuned)** | **0.7445** | **0.5117** | **0.8155** | **0.6289** | **0.8463** | **0.6531** |

## Hyperparameter Tuning

**Method:** RandomizedSearchCV, `n_iter=20`, 5-fold CV, `scoring='recall'` (business priority).

**Search space:**

| Parameter | Range |
|---|---|
| `n_estimators` | 100-600 |
| `max_depth` | 3-12 |
| `learning_rate` | 0.01-0.21 |
| `subsample` | 0.6-1.0 |
| `colsample_bytree` | 0.6-1.0 |
| `min_child_weight` | 1-10 |
| `gamma` | 0-0.5 |

**Best parameters:**

n_estimators=149, max_depth=3, learning_rate=0.017,
subsample=0.865, colsample_bytree=0.649,
min_child_weight=4, gamma=0.248



**Best CV recall:** 0.8140 (test-set recall 0.8155 — consistent, no overfitting to test).

## Confusion Matrix (Test Set, threshold=0.5)

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 743 | 292 |
| **Actual Yes** | 69 | 305 |

- **True Positives (caught churners):** 305
- **False Negatives (missed churners):** 69
- **False Positives (false alarms):** 292
- **True Negatives (correctly cleared):** 743

**Interpretation:** Of 374 actual churners, 305 are caught (81.5% recall). Of 1,035 actual non-churners, 292 are incorrectly flagged (28.2% false positive rate). This tradeoff is intentional — false alarms cost less than missing churners.

## Calibration

The model uses `scale_pos_weight` and a probability threshold of 0.5. No explicit calibration has been applied. For production use where probability magnitude matters (e.g., ranking by expected loss), consider adding `CalibratedClassifierCV` or applying isotonic regression post-hoc.

## Error Analysis

### False Negatives (missed churners — 69 customers)

Analysis of missed churners shows they cluster around:
- **Two-year contract customers with short tenure** — the model under-weights this combination; usually contract type dominates, but short tenure on a long contract should still be a risk signal.
- **Low monthly charges with many services** — the model may be reading "many services" as engagement and missing the low-revenue signal.

**Systematic weakness:** The model under-predicts churn for month-to-month customers whose only negative signals are tenure and service count, without high charges.

### False Positives (false alarms — 292 customers)

- **Month-to-month + high charges** — this is the model's strongest positive pattern, and it over-fires.
- **Long-tenure month-to-month customers** — tenure typically protects, but `ContractRisk` dominates in borderline cases.

**Retention team should prioritize false positives carefully:** contacting all 292 will waste effort, but filtering by manual review (probability 0.5–0.7 only) reduces this to ~180.

## Business Impact (Illustrative)

Assuming:
- Missed churner = $500 loss
- Retention offer cost = $50
- Offer success rate = 30%

| Model | Missed (FN) | Offers sent (TP+FP) | Estimated loss | Estimated cost | Net |
|---|---|---|---|---|---|
| Baseline | 374 | 0 | $187,000 | $0 | -$187,000 |
| **XGBoost tuned** | 69 | 597 | $34,500 | $29,850 | -$64,350 |

Net improvement: **~$122,650 saved** versus baseline (on 1,409 test customers).

## Reproducibility

- All seeds set to `random_state=42`
- Pipeline and preprocessing fit inside CV folds (no leakage)
- Model artifacts saved with date-tagged filenames
- Full training reproducible via `python -m src.train`