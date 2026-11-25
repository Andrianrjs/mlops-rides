import json
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Rides Classifier",
    page_icon="🚲",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.title("🚲 Rides Behavior Classification")
st.markdown(
    "Classify riding behavior from IMU time series (accelerometer & gyroscope) into "
    "**Safe**, **Safe Overall**, **Bad Weather**, **Tandem**, or **Reckless**."
)

# Sidebar: API Health Status
st.sidebar.header("Backend API Status")
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success(f"🟢 API Online\nModel: {health_data.get('model_version', 'loaded')}")
    else:
        st.sidebar.warning(f"🟡 API Degraded: {health_resp.json().get('detail', {}).get('error', 'No model')}")
except Exception as e:
    st.sidebar.error(f"🔴 API Offline ({e})")

st.sidebar.markdown("---")
input_option = st.sidebar.radio(
    "Choose Ride Input",
    ["Select Predefined Sample", "Upload Custom Ride (CSV/JSON)"]
)

COLUMNS = ["ax", "ay", "az", "rx", "ry", "rz"]

def generate_sample(sample_type: str) -> np.ndarray:
    t = np.linspace(0, 10, 100)  # 100 timesteps (contract T=100)
    if sample_type == "Safe Ride (Normal)":
        ax = 0.05 * np.sin(t) + np.random.normal(0, 0.02, size=len(t))
        ay = 0.03 * np.cos(t) + np.random.normal(0, 0.02, size=len(t))
        az = 9.81 + np.random.normal(0, 0.05, size=len(t))
        rx = np.random.normal(0, 0.01, size=len(t))
        ry = np.random.normal(0, 0.01, size=len(t))
        rz = np.random.normal(0, 0.01, size=len(t))
    elif sample_type == "Reckless Ride (Aggressive Swerving/Acceleration)":
        ax = 4.0 * np.sin(2 * t) + np.random.normal(0, 0.5, size=len(t))
        ay = 3.5 * np.cos(2 * t) + np.random.normal(0, 0.5, size=len(t))
        az = 9.81 + 2.0 * np.sin(5 * t) + np.random.normal(0, 0.8, size=len(t))
        rx = 1.2 * np.sin(3 * t) + np.random.normal(0, 0.2, size=len(t))
        ry = 1.5 * np.cos(3 * t) + np.random.normal(0, 0.2, size=len(t))
        rz = 0.8 * np.sin(t) + np.random.normal(0, 0.2, size=len(t))
    else:  # Bad Weather / Rough Surface
        ax = np.random.normal(0, 0.25, size=len(t))
        ay = np.random.normal(0, 0.25, size=len(t))
        az = 9.81 + np.random.normal(0, 0.4, size=len(t))
        rx = np.random.normal(0, 0.08, size=len(t))
        ry = np.random.normal(0, 0.08, size=len(t))
        rz = np.random.normal(0, 0.08, size=len(t))
    return np.column_stack([ax, ay, az, rx, ry, rz])

readings_df = None

if input_option == "Select Predefined Sample":
    sample_choice = st.selectbox(
        "Select Ride Scenario",
        [
            "Safe Ride (Normal)",
            "Bad Weather / Rough Surface",
            "Reckless Ride (Aggressive Swerving/Acceleration)"
        ]
    )
    data = generate_sample(sample_choice)
    readings_df = pd.DataFrame(data, columns=COLUMNS)

else:
    uploaded_file = st.file_uploader("Upload ride file (CSV with columns ax,ay,az,rx,ry,rz or JSON)", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                readings_df = pd.read_csv(uploaded_file)
                if not all(col in readings_df.columns for col in COLUMNS):
                    st.error(f"CSV must contain columns: {COLUMNS}")
                    readings_df = None
                else:
                    readings_df = readings_df[COLUMNS]
            else:
                content = json.load(uploaded_file)
                if isinstance(content, dict) and "readings" in content:
                    readings_df = pd.DataFrame(content["readings"], columns=COLUMNS)
                elif isinstance(content, list):
                    readings_df = pd.DataFrame(content, columns=COLUMNS)
        except Exception as err:
            st.error(f"Error parsing uploaded file: {err}")

if readings_df is not None:
    st.subheader("📊 Ride IMU Signals Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Accelerometer ($m/s^2$)**")
        st.line_chart(readings_df[["ax", "ay", "az"]])
    with col2:
        st.markdown("**Gyroscope ($rad/s$)**")
        st.line_chart(readings_df[["rx", "ry", "rz"]])

    st.write(f"**Shape:** `{readings_df.shape[0]} timesteps` × `{readings_df.shape[1]} channels`")

    if st.button("🚀 Classify Ride", type="primary"):
        with st.spinner("Calling Prediction API..."):
            try:
                payload = {"readings": readings_df.values.tolist()}
                resp = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=5)
                
                if resp.status_code == 200:
                    result = resp.json()
                    label = result["label"]
                    probas = result["probabilities"]
                    model_ver = result.get("model_version", "N/A")

                    st.markdown("---")
                    st.header("Prediction Result")

                    if label == "Reckless":
                        st.error(f"🚨 **Class: {label}** (High Risk Detected)")
                    elif label in ["Safe", "Safe Overall"]:
                        st.success(f"✅ **Class: {label}**")
                    else:
                        st.warning(f"⚠️ **Class: {label}**")

                    st.caption(f"Model Run ID / Version: `{model_ver}`")

                    st.subheader("Class Probabilities")
                    prob_df = pd.DataFrame(
                        list(probas.items()),
                        columns=["Class", "Probability"]
                    ).set_index("Class")
                    st.bar_chart(prob_df)

                elif resp.status_code == 503:
                    st.warning("⚠️ API is running, but no MLflow model is loaded yet.")
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Ensure `uv run uvicorn rides.api.main:app` is running on port 8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
