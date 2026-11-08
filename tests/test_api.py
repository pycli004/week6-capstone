import os
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from backend.main import app

# Load .env FIRST, before reading API_KEY
load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set in .env file")

client = TestClient(app)

VALID_INPUT = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_happy_path():
    r = client.post("/predict", json=VALID_INPUT, headers={"x-api-key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in ["Churn", "No Churn"]
    assert len(data["top_drivers"]) == 3


def test_missing_field():
    bad = VALID_INPUT.copy()
    del bad["tenure"]
    r = client.post("/predict", json=bad, headers={"x-api-key": API_KEY})
    assert r.status_code == 422


def test_invalid_type():
    bad = VALID_INPUT.copy()
    bad["tenure"] = "not-a-number"
    r = client.post("/predict", json=bad, headers={"x-api-key": API_KEY})
    assert r.status_code == 422


def test_auth_failure():
    r = client.post("/predict", json=VALID_INPUT, headers={"x-api-key": "wrong-key"})
    assert r.status_code == 401


def test_regression_known_input():
    r1 = client.post("/predict", json=VALID_INPUT, headers={"x-api-key": API_KEY})
    r2 = client.post("/predict", json=VALID_INPUT, headers={"x-api-key": API_KEY})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["churn_probability"] == r2.json()["churn_probability"]
    assert r1.json()["churn_prediction"] == r2.json()["churn_prediction"]