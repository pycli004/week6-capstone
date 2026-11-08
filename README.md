ChurnGuard --- Customer Churn Prediction System
A machine learning system that predicts customer churn risk and explains
the top contributing factors, built for telecom retention teams.
Features
- Tuned XGBoost classifier with 5-fold cross-validated performance
- SHAP-based explanations:
  - Global summary
  - Top 3 contributing factors for each customer
- FastAPI backend with API key authentication and structured logging
- Streamlit frontend for interactive predictions
- Automated pytest suite covering:
  - Happy path
  - Authentication
  - Validation
  - Regression
Project Structure
churnguard/
├── backend/       # FastAPI service
├── frontend/      # Streamlit UI
├── src/           # Feature engineering, training, SHAP
├── models/        # Trained model artifacts (.joblib, .skops)
├── data/          # Telco churn dataset
├── tests/         # Pytest suite
├── logs/          # Prediction logs (auto-generated)
├── docs/          # Extended documentation
├── requirements.txt

Setup
1. Clone the repository and install dependencies
git clone <repo-url>
cd churnguard

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
2. Configure environment variables
Create a `.env` file in the project root and set your API key:
API_KEY=your-secret-key-here
BACKEND_URL=http://127.0.0.1:8000
3. Train the model (first time only)
python -m src.train
python -m src.explain
This creates:
- models/churnguard_xgb_<date>.joblib
- models/shap_summary.png
Run the Application
Backend (Terminal 1)
python -m uvicorn backend.main:app --reload --port 8000
- API: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
Frontend (Terminal 2)
python -m streamlit run frontend/app.py
- UI: http://localhost:8501
Run Tests
python -m pytest tests/ -v
API Reference
GET /health
Returns the backend status and the loaded model path.
POST /predict
Requires the following request header:
x-api-key: <API_KEY>
Request body: customer profile containing 19 fields matching the Telco
dataset schema.

Example response:
{
  "churn_probability": 0.853,
  "churn_prediction": "Churn",
  "top_drivers": [
    {
      "feature": "num__ContractRisk",
      "impact": 0.42,
      "direction": "increases"
    },
    {
      "feature": "num__tenure",
      "impact": -0.31,
      "direction": "decreases"
    },
    {
      "feature": "num__MonthlyCharges",
      "impact": 0.18,
      "direction": "increases"
    }
  ]
}
Documentation
- [Model Card](docs/MODEL_CARD.md) --- Model characteristics, intended
  use, and limitations
- [Architecture](docs/ARCHITECTURE.md) --- System components and data
  flow
- [Evaluation Report](docs/EVALUATION_REPORT.md) --- Full metrics,
  confusion matrix, and error analysis
- [Monitoring Notes](docs/MONITORING_NOTES.md) --- Drift detection and
  retraining plan
Environment Variables
  Variable                Description             Default
  API_KEY               Required for /predict None --- must be set
BACKEND_URL           Backend URL for the     http://127.0.0.1:8000
frontend
Notes
- The model is trained on the Telco Customer Churn
  dataset.
- All preprocessing is fitted only on training data to prevent data
  leakage.
- Prediction logs are written to logs/predictions.csv.
Model Card --- ChurnGuard Churn Classifier
Model Details
- Model type: XGBoost classifier (gradient-boosted trees)
- Version: 1.0
- Training date: See filename
  models/churnguard_xgb_<date>.joblib
- Framework: scikit-learn Pipeline + XGBoost
- Owner: Internship ML track
Intended Use
Intended
- Predict the probability that a telecom customer will churn.
- Support retention teams in prioritizing customer outreach.
- Provide interpretable per-customer explanations using the top 3
  contributing factors.
Not Intended
- Automated decision-making without human review.
- Pricing or penalty decisions.
- Use on non-telecom customer data without retraining.
- Legal, financial, or employment decisions.
Training Data
- Source: Telco Customer Churn (Kaggle --- IBM sample dataset)
- Rows: 7,043 customers
- Features: 19 raw features + 5 engineered features = 24 columns
- Target: Churn (binary, imbalanced --- approximately 27%
  positive)
- Split: Stratified 80/20 train/test split (random_state=42,
  stratify=y)
Engineered Features
  Feature                             Description
  TenureBucket                      Customer tenure grouped into 4
buckets (0--1 year, 1--2 years,
2--4 years, 4+ years)

  TotalServices                     Count of subscribed services (0--9)

  ChargesPerTenure                  MonthlyCharges / (tenure + 1) ---
value-for-money signal

  ContractRisk                      1 if the contract is
month-to-month; otherwise 0
HasFamily                         1 if the customer has a Partner or
Dependents
Evaluation Metrics
Test set: 1,409 customers.

  Metric                 Value
  Accuracy              0.7445
Precision (Churn)     0.5117
Recall (Churn)        0.8155
F1 (Churn)            0.6289
ROC-AUC               0.8463
PR-AUC                0.6531

Comparison with the majority-class baseline:
- Baseline accuracy: 0.7346
- Baseline churn recall: 0.0000
- Tuned model churn recall: 0.8155
The baseline catches zero churners, while the tuned model catches
approximately 81.5% of churners in the test set.
Limitations
- Population: Trained on US telecom data from the 2010s. It may
  not generalize to other markets or current customer bases.
- Temporal drift: No time-series validation was performed.
  Customer behavior in production may change over time.
- Label quality: The churn label reflects historical behavior, not
  customer intent. Some customers marked as retained may churn after
  the dataset snapshot.
- Feature set: Excludes account-level value (LTV, ARPU), support
  ticket history, and customer interactions --- signals a real
  production system would likely include.
- Threshold: The classification threshold is fixed at 0.5. A
  business-specific threshold (for example, 0.3 to increase recall)
  may be more appropriate and would change the reported metrics.
- Interpretation: SHAP explanations describe model behavior; they
  do not establish causal relationships.
Ethical Considerations
- Do not use the model to deny service or discriminate through
  pricing.
- Retention offers based on churn risk should be offered fairly across
  customer groups.
- Log all predictions (implemented in logs/predictions.csv) to
  support audits.
Retraining
- Retrain quarterly or when data drift is detected (see
  MONITORING_NOTES.md).
- Keep the same pipeline and feature engineering process for
  consistency.
- Validate the model against a fresh hold-out set before promoting a
  new version.