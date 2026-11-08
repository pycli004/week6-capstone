# ChurnGuard — Architecture

## System Overview

ChurnGuard is a three-tier system:
[Streamlit UI] ──HTTP──> [FastAPI Backend] ──load──> [XGBoost Model]
│
└──append──> [logs/predictions.csv]


## Components

### 1. Model Layer (`src/`, `models/`)

- `src/features.py` — Feature engineering (5 engineered features)
- `src/train.py` — Training pipeline, cross-validation, tuning, model save
- `src/explain.py` — SHAP global and local explanations
- `models/churnguard_xgb_<date>.joblib` — Serialized pipeline (preprocessor + classifier)
- `models/churnguard_xgb_<date>.skops` — Skops-secure alternative

**Pipeline (serialized as one object):**

Raw customer data
│
▼
engineer_features() (TenureBucket, TotalServices, ChargesPerTenure, ContractRisk, HasFamily)
│
▼
ColumnTransformer
├── Numeric: SimpleImputer(median) → StandardScaler
└── Categorical: SimpleImputer("None") → OneHotEncoder
│
▼
XGBClassifier (tuned via RandomizedSearchCV)
│
▼
churn_probability


### 2. Backend Layer (`backend/`)

**Framework:** FastAPI

**Files:**
- `backend/main.py` — App, routes, auth, logging
- `backend/schemas.py` — Pydantic request/response models

**Endpoints:**

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | None | Liveness check, returns model path |
| POST | `/predict` | `x-api-key` | Churn prediction + SHAP drivers |

**Authentication:** Header `x-api-key` must match `API_KEY` in `.env`.

**Request flow for `/predict`:**

1. Receive `CustomerInput` (validated by Pydantic)
2. Convert to one-row DataFrame
3. Apply `engineer_features`
4. Transform via fitted preprocessor (from model pipeline)
5. Predict probability
6. Compute SHAP values for the single customer
7. Extract top 3 drivers by absolute SHAP magnitude
8. Append to `logs/predictions.csv`
9. Return `PredictionOutput`

**Logging format** (`logs/predictions.csv`):
timestamp,probability,prediction,top_driver_1,top_driver_2,top_driver_3
2025-11-08T10:30:00Z,0.853,Churn,num__ContractRisk,num__tenure,num__MonthlyCharges


### 3. Frontend Layer (`frontend/`)

**Framework:** Streamlit

**File:** `frontend/app.py`

**Flow:**

1. Sidebar shows backend health
2. Form with 19 customer fields in 3 columns
3. On submit, POSTs to backend `/predict`
4. Displays:
   - Churn probability (red if high, green if low)
   - Classification label
   - Top 3 drivers with direction and impact
   - Raw JSON in expander

**No server-side state.** Frontend is stateless; each prediction is independent.

## Data Flow
Customer form submit
│
▼
Streamlit → POST http://localhost:8000/predict
headers: {x-api-key}
body: {19 customer fields}
│
▼
FastAPI validates → model.predict_proba
│
├── SHAP TreeExplainer (local values for this customer)
│
└── append to logs/predictions.csv
│
▼
Streamlit renders result


## Environment & Secrets

- `.env` file holds `API_KEY` and `BACKEND_URL`
- `.env.example` provides the schema (safe to commit)
- `.env` is git-ignored
- No secrets in code

## Testing

- `tests/test_api.py` — 6 API tests (health, happy path, missing field, invalid type, auth failure, regression)
- `tests/test_features.py` — 2 feature engineering tests

Run with `python -m pytest tests/ -v`.

## Deployment (Local)

- Backend: `python -m uvicorn backend.main:app --reload --port 8000`
- Frontend: `python -m streamlit run frontend/app.py`
- Both require the venv activated

## Future Enhancements (Out of Scope)

- Dockerized deployment
- Public cloud hosting (Hugging Face Spaces / Render)
- Real-time drift monitoring
- Batch prediction endpoint
- Retraining pipeline triggered by drift