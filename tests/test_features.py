import pandas as pd
from src.features import engineer_features


def test_feature_engineering_creates_columns():
    
    df = pd.DataFrame([{
        'tenure': 5, 'MonthlyCharges': 50.0, 'TotalCharges': 250.0,
        'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No', 'DeviceProtection': 'No',
        'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'Partner': 'No', 'Dependents': 'No',
    }])
    out = engineer_features(df)
    for col in ['TenureBucket', 'TotalServices', 'ChargesPerTenure', 'ContractRisk', 'HasFamily']:
        assert col in out.columns


def test_contract_risk_flag():
    
    df = pd.DataFrame([{
        'tenure': 5, 'MonthlyCharges': 50.0, 'TotalCharges': 250.0,
        'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No', 'DeviceProtection': 'No',
        'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'Partner': 'No', 'Dependents': 'No',
    }])
    out = engineer_features(df)
    assert out['ContractRisk'].iloc[0] == 1