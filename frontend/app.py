
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY")

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)

if not API_KEY:
    st.error("API_KEY not set in .env file")
    st.stop()

st.title("ChurnGuard")
st.caption("Customer Churn Prediction & Retention Support System")

st.divider()

with st.sidebar:
    st.header("System Status")

    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)

        if r.status_code == 200:
            st.success("Backend Online")

            with st.expander("Backend Details"):
                st.json(r.json())
        else:
            st.error(f"Backend Error: {r.status_code}")

    except requests.RequestException:
        st.error("Backend Offline")

    st.divider()
    st.caption("ChurnGuard v1.0")
    st.caption("Customer Retention Analytics")

st.subheader("Customer Details")
st.caption("Enter customer information to predict churn probability.")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Personal Information")

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=1
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col2:
        st.markdown("#### Internet Services")

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        st.markdown("#### Billing & Contract")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=29.85,
            step=0.01
        )
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=29.85,
            step=0.01
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

if submitted:

    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total)
    }

    try:
        with st.spinner("Analyzing customer profile..."):

            r = requests.post(
                f"{BACKEND_URL}/predict",
                json=payload,
                headers={"x-api-key": API_KEY},
                timeout=10
            )

        if r.status_code == 200:

            data = r.json()
            proba = float(data["churn_probability"])
            label = data["churn_prediction"]

            st.divider()
            st.subheader("Prediction Results")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Churn Probability",
                    f"{proba:.1%}"
                )

            with c2:
                st.metric(
                    "Prediction",
                    label
                )

            if label == "Churn":
                st.error("High Risk: Consider a customer retention offer.")
            else:
                st.success("Low Risk: Customer is predicted to stay.")

            st.progress(
                max(0.0, min(1.0, proba)),
                text="Churn Probability"
            )

            st.subheader("Top 3 Contributing Factors")

            for i, d in enumerate(data["top_drivers"], 1):

                feature = (
                    d["feature"]
                    .replace("num__", "")
                    .replace("cat__", "")
                )

                direction = d["direction"]

                if direction == "increases":
                    st.warning(
                        f"{i}. {feature} — Increases churn risk "
                        f"(Impact: {d['impact']:+.3f})"
                    )
                else:
                    st.info(
                        f"{i}. {feature} — Decreases churn risk "
                        f"(Impact: {d['impact']:+.3f})"
                    )

            with st.expander("Raw API Response"):
                st.json(data)

        elif r.status_code == 401:
            st.error("Authentication failed. Check API_KEY in .env.")

        elif r.status_code == 422:
            st.error(f"Validation error: {r.json()}")

        else:
            st.error(f"Error {r.status_code}: {r.text}")

    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

st.divider()
st.caption("ChurnGuard | Customer Retention Intelligence | Version 1.0")
