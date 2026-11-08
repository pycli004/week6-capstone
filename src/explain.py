import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.train import load_and_split


def load_model():

    import glob
    model_path = sorted(glob.glob('models/churnguard_xgb_*.joblib'))[-1]
    return joblib.load(model_path), model_path


def main():
    
    X_train, X_test, y_train, y_test = load_and_split()
    model, path = load_model()
    print(f"Loaded model: {path}")

    
    preprocessor = model.named_steps['prep']
    classifier = model.named_steps['clf']

    
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_transformed,
        feature_names=feature_names, show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig('models/shap_summary.png', bbox_inches='tight', dpi=100)
    plt.close()
    print("Saved: models/shap_summary.png")

    
    print("\n=== Local explanations (top 3 churn-risk customers) ===")
    proba = classifier.predict_proba(X_test_transformed)[:, 1]
    top_risk_idx = np.argsort(proba)[-3:][::-1]

    for i, idx in enumerate(top_risk_idx, 1):
        print(f"\n--- Customer {i} (churn probability: {proba[idx]:.2%}) ---")
        customer_shap = shap_values[idx]
        top_features_idx = np.argsort(np.abs(customer_shap))[-3:][::-1]

        for fi in top_features_idx:
            fname = feature_names[fi]
            fval = X_test_transformed[idx, fi]
            contrib = customer_shap[fi]
            direction = "increases" if contrib > 0 else "decreases"
            print(f"  {fname} = {fval:.2f}  →  {direction} churn risk by {abs(contrib):.3f}")


if __name__ == "__main__":
    main()