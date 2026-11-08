import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from src.features import engineer_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


def load_and_split():
    
    df = pd.read_csv('data/telco_churn.csv')
    df = engineer_features(df)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)


    y = (df['Churn'] == 'Yes').astype(int)

    
    X = df.drop(['customerID', 'Churn'], axis=1)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(name, y_true, y_pred, y_proba=None):

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
        print(f"PR-AUC:    {average_precision_score(y_true, y_proba):.4f}")


if __name__ == "__main__":

    
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate train: {y_train.mean():.3f}")
    print(f"Churn rate test:  {y_test.mean():.3f}")

    
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    y_proba = baseline.predict_proba(X_test)[:, 1]
    evaluate_model("Baseline (Majority)", y_test, y_pred, y_proba)

        
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"\nNumeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")

    
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    
    lr_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    evaluate_model("Logistic Regression (balanced)", y_test, y_pred_lr, y_proba_lr)

        
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    evaluate_model("Random Forest (balanced)", y_test, y_pred_rf, y_proba_rf)

    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, verbosity=0,
            eval_metric='logloss'
        ))
    ])
    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)
    y_proba_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost (scale_pos_weight)", y_test, y_pred_xgb, y_proba_xgb)


        
    print("\n" + "="*50)
    print("TUNING MODELS (this takes a few minutes)")
    print("="*50)


    xgb_param_dist = {
        'clf__n_estimators': randint(100, 600),
        'clf__max_depth': randint(3, 12),
        'clf__learning_rate': uniform(0.01, 0.2),
        'clf__subsample': uniform(0.6, 0.4),
        'clf__colsample_bytree': uniform(0.6, 0.4),
        'clf__min_child_weight': randint(1, 10),
        'clf__gamma': uniform(0, 0.5),
    }

    xgb_search = RandomizedSearchCV(
        Pipeline([('prep', preprocessor), ('clf', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss'
        ))]),
        param_distributions=xgb_param_dist,
        n_iter=20,
        cv=5,
        scoring='recall',   
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    import time
    start = time.time()
    xgb_search.fit(X_train, y_train)
    print(f"\nXGBoost tuning time: {time.time() - start:.1f}s")
    print(f"Best CV recall: {xgb_search.best_score_:.4f}")
    print(f"Best params: {xgb_search.best_params_}")

    
    y_pred_xgb_tuned = xgb_search.best_estimator_.predict(X_test)
    y_proba_xgb_tuned = xgb_search.best_estimator_.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost (TUNED)", y_test, y_pred_xgb_tuned, y_proba_xgb_tuned)


    
    import pandas as pd
    results = pd.DataFrame([
        ['Baseline (Majority)', 0.7346, 0.0000, 0.0000, 0.0000, 0.5000, 0.2650],
        ['Logistic Regression', 0.7367, 0.5026, 0.7727, 0.6091, 0.8461, 0.6645],
        ['Random Forest', 0.7587, 0.5324, 0.7460, 0.6214, 0.8386, 0.6478],
        ['XGBoost (original)', 0.7601, 0.5340, 0.7567, 0.6261, 0.8355, 0.6448],
        ['XGBoost (tuned)', 0.7445, 0.5117, 0.8155, 0.6289, 0.8463, 0.6531],
    ], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'])

    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(results.to_string(index=False))
    results.to_csv('models/model_comparison.csv', index=False)
    print("\nSaved to models/model_comparison.csv")



    import joblib
    from datetime import datetime

    date_str = datetime.now().strftime('%Y-%m-%d')
    model_path = f'models/churnguard_xgb_{date_str}.joblib'
    joblib.dump(xgb_search.best_estimator_, model_path)
    print(f"\nBest model saved to: {model_path}")

    
    import skops.io as sio
    skops_path = f'models/churnguard_xgb_{date_str}.skops'
    sio.dump(xgb_search.best_estimator_, skops_path)
    print(f"Skops model saved to: {skops_path}")