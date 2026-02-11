import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px

st.set_page_config(page_title="AI Player Behaviour Analyzer", layout="wide")

st.title("🎮 AI Player Behaviour Pattern Analyzer")
st.write("Deep Learning Based Reduced Usage & Stress Prediction System")

# Load Model + Preprocessing Files
@st.cache_resource
def load_files():
    model = load_model("lstm_model.keras")
    scaler = joblib.load("scaler (1).pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

model, scaler, encoders = load_files()

# Upload Dataset
uploaded_file = st.file_uploader("Upload Gameplay Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    categorical_cols = ['Gender','Location','GameGenre','GameDifficulty','EngagementLevel']

    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    # Feature selection (same as training phase)
    X = df.drop("PlayerID", axis=1)

    # Scaling
    X_scaled = scaler.transform(X)

    # Reshape for LSTM
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Prediction
    predictions = model.predict(X_lstm)

    df["Reduced_Usage_Risk"] = predictions[:, 0]
    df["Stress_Risk"] = predictions[:, 1]

    # Convert to percentage
    df["Reduced_Usage_Risk"] = (df["Reduced_Usage_Risk"] * 100).round(2)
    df["Stress_Risk"] = (df["Stress_Risk"] * 100).round(2)

    st.subheader("🔮 Prediction Results")
    st.dataframe(df[["PlayerID","Reduced_Usage_Risk","Stress_Risk"]])

    # ================= DASHBOARD =================

    st.subheader("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="Reduced_Usage_Risk",
                            title="Reduced Usage Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x="Stress_Risk",
                            title="Stress Risk Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.success("Prediction Completed Successfully ✅")

else:
    st.info("Please upload a dataset to begin analysis.")
