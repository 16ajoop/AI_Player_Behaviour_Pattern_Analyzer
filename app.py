import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Player Behaviour Analyzer",
                   layout="wide",
                   page_icon="🎮")

# ---------------- VIOLET THEME ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    color: #ffffff;
}

/* Titles */
h1, h2, h3, h4 {
    color: #c77dff !important;
    font-weight: 700;
}

/* Upload box */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #7b2cbf;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 10px;
}

/* Buttons */
.stDownloadButton > button {
    background: linear-gradient(90deg,#7b2cbf,#c77dff);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    padding: 10px 20px;
}

.stDownloadButton > button:hover {
    background: linear-gradient(90deg,#9d4edd,#e0aaff);
    transform: scale(1.05);
}

/* Success message */
.stAlert {
    border-radius: 12px;
}

/* Charts container glow */
.element-container {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(123,44,191,0.3);
}

/* Scroll bar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #7b2cbf;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    cls_model = load_model("behaviour_classifier.keras")
    risk_model = load_model("player_behavior_lstm.keras")
    return scaler, feature_names, cls_model, risk_model

scaler, feature_names, cls_model, risk_model = load_models()

# ---------------- HEADER ----------------
st.title("🎮 AI Player Behaviour Dashboard")
uploaded_file = st.file_uploader("Upload Player Dataset CSV", type=["csv"])

# ---------------- FEATURE ENGINEERING FUNCTION ----------------
def engineer_features(df):
    df['EngagementScore'] = df['PlayTimeHours'] * df['SessionsPerWeek']
    df['ActivityConsistency'] = df['SessionsPerWeek'] / 7
    df['PerformanceScore'] = (df['PlayerLevel'] + df['AchievementsUnlocked']) / 2
    df['Achievement_Rate'] = df['AchievementsUnlocked'] / (df['PlayerLevel'] + 1)
    df['AddictionRisk'] = (df['PlayTimeHours'] > 40).astype(int)
    df['StressIndicator'] = (df['AvgSessionDurationMinutes'] > 120).astype(int)
    df['ReducedUsageTrend'] = (df['SessionsPerWeek'] < 3).astype(int)

    df["PlayTime_per_Session"] = df["PlayTimeHours"] / (df["SessionsPerWeek"] + 1)
    df["Purchase_per_Hour"] = df["InGamePurchases"] / (df["PlayTimeHours"] + 1)
    return df

# ---------------- MAIN APP ----------------
if uploaded_file is not None:

    df_original = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df_original)

    df = df_original.copy()

    # Save Player IDs
    if "PlayerID" in df.columns:
        player_ids = df["PlayerID"]
        df = df.drop("PlayerID", axis=1)
    else:
        player_ids = np.arange(len(df))

    # 🔥 FEATURE ENGINEERING (FIX)
    df = engineer_features(df)
    df.fillna(0, inplace=True)

    # 🔥 MATCH TRAINING FEATURES (FIX)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    # Reshape for LSTM
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # ---------------- PREDICTIONS ----------------
    cls_pred = cls_model.predict(X_lstm)
    risk_pred = risk_model.predict(X_lstm)

    behaviour_map = {0:"Casual",1:"Regular",2:"Addicted"}
    behaviour_labels = [behaviour_map[np.argmax(i)] for i in cls_pred]

    reduced_usage = (risk_pred[:,0]*100).round(2)
    stress_risk = (risk_pred[:,1]*100).round(2)

    results = pd.DataFrame({
        "PlayerID": player_ids,
        "Behaviour": behaviour_labels,
        "ReducedUsageRisk%": reduced_usage,
        "StressRisk%": stress_risk
    })

    # ---------------- DASHBOARD ----------------
    st.header("🤖 Prediction Results")
    st.dataframe(results, use_container_width=True)

    # ---------------- LEADERBOARD ----------------
    st.header("🏆 Risk Leaderboard (Top 10 Players)")
    leaderboard = results.sort_values("StressRisk%", ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

    # ---------------- CHARTS ----------------
    col1,col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(results, x="StressRisk%", title="Stress Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(results, names="Behaviour", title="Behaviour Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # Gauge
    st.header("🎯 Overall Addiction Gauge")
    avg_risk = results["StressRisk%"].mean()
    fig = go.Figure(go.Indicator(mode="gauge+number",
                                 value=avg_risk,
                                 title={'text':"Average Stress Risk"},
                                 gauge={'axis':{'range':[0,100]}}))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- DOWNLOAD BUTTON ----------------
    st.download_button(
        "📥 Download Full Prediction Report",
        results.to_csv(index=False),
        "Player_Behaviour_Report.csv",
        "text/csv"
    )

    st.success("Analysis Completed Successfully 🎉")

else:
    st.info("Upload dataset to start analysis")
