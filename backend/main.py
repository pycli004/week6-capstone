import os
from dotenv import load_dotenv
load_dotenv(override=True)

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, Depends
from backend.schemas import CustomerInput, PredictionOutput
from src.features import engineer_features


# --- Config: load API key from .env only (no default) ---
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set in .env file")

MODEL_PATH = sorted(Path("models").glob("churnguard_xgb_*.joblib"))[-1]
LOG_PATH = Path("logs/predictions.csv")

app = FastAPI(title="ChurnGuard API", version="1.0")

# Load model once at startup
model = joblib.load(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def log_prediction(customer: dict, proba: float, label: str, drivers: list):
    LOG_PATH.parent.mkdir(exist_ok=True)
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'probability': round(proba, 4),
        'prediction': label,
        'top_driver_1': drivers[0]['feature'] if len(drivers) > 0 else '',
        'top_driver_2': drivers[1]['feature'] if len(drivers) > 1 else '',
        'top_driver_3': drivers[2]['feature'] if len(drivers) > 2 else '',
    }
    df_row = pd.DataFrame([row])
    if LOG_PATH.exists():
        df_row.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode='w', header=True, index=False)


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput, api_key: str = Depends(verify_api_key)):
    # Convert input to one-row DataFrame
    df = pd.DataFrame([customer.model_dump()])
    df['customerID'] = 'api-input'
    df['Churn'] = 'No'

    # Engineer features
    df = engineer_features(df)

    # Drop ID and target
    X = df.drop(['customerID', 'Churn'], axis=1)

    # Predict
    proba = float(model.predict_proba(X)[0, 1])
    label = "Churn" if proba >= 0.5 else "No Churn"

    # SHAP local explanation
    preprocessor = model.named_steps['prep']
    classifier = model.named_steps['clf']
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    import shap
    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(X_transformed)[0]
    top_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]

    drivers = []
    for i in top_idx:
        drivers.append({
            "feature": feature_names[i],
            "impact": round(float(shap_vals[i]), 4),
            "direction": "increases" if shap_vals[i] > 0 else "decreases"
        })

    # Log prediction
    log_prediction(customer.model_dump(), proba, label, drivers)

    return PredictionOutput(
        churn_probability=round(proba, 4),
        churn_prediction=label,
        top_drivers=drivers
    )