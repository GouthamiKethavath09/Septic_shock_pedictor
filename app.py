import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# ---------------- BACKGROUND + GLASS ---------------- #
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}

    .glass {{
        background: rgba(255,255,255,0.15);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
    }}

    h1,h2,h3,h4,p,li {{
        color: white !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: black !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("img.jpg")

# ---------------- CONFIG ---------------- #
SEQ_LENGTH = 24
DEFAULT_AGE = 55

FEATURE_NAMES = [
    "bp","creatinine","heart_rate",
    "lactate","resp_rate","temperature","wbc","age"
]

# ---------------- LOAD ---------------- #
model = load_model("septic_model.h5", compile=False)
scaler = pickle.load(open("Notebook/scaler.pkl", "rb"))

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align:center;'>🧠 Septic Shock AI Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📌 System Overview")
st.sidebar.info("""
🔬 AI-powered ICU system

✔ Predicts septic shock risk  
✔ Analyzes patient vitals  
✔ Provides medical insights  
✔ Suggests precautions  

📊 Features Used:
- Blood Pressure (BP)
- Creatinine
- Heart Rate
- Lactate
- Respiration Rate
- Temperature
- WBC Count
- Age
""")

# ---------------- UPLOAD ---------------- #
st.markdown("## 📂 Upload Patient Data")
file = st.file_uploader("Upload CSV (24×7)", type=["csv"])

data_array = None

if file:
    df = pd.read_csv(file)

    if df.shape != (24,7):
        st.error("❌ CSV must be 24 rows × 7 columns")
    else:
        st.success("✅ Data Loaded")
        st.dataframe(df)
        data_array = df.values

# ---------------- PREDICT ---------------- #

if st.button("🚀 Analyze Patient"):

    if data_array is None:
        st.warning("Upload data first")

    else:
        age_col = np.full((SEQ_LENGTH,1), DEFAULT_AGE)
        data = np.hstack([data_array, age_col])

        data_scaled = scaler.transform(data)
        data_scaled = data_scaled.reshape(1,24,8)

        pred = model.predict(data_scaled)[0][0]

        # ---------------- TOP METRICS ---------------- #
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='glass'><h3>Risk Score</h3></div>", unsafe_allow_html=True)
            st.metric("Probability", f"{pred:.2f}")

        with col2:
            if pred > 0.7:
                status = "HIGH RISK"
            elif pred > 0.4:
                status = "MODERATE"
            else:
                status = "LOW RISK"
            st.markdown("<div class='glass'><h3>Status</h3></div>", unsafe_allow_html=True)
            st.metric("Condition", status)

        with col3:
            st.markdown("<div class='glass'><h3>Confidence</h3></div>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{pred*100:.1f}%")
def show_comparison(df):
    normal = {
        "bp": 120,
        "heart_rate": 75,
        "lactate": 1.0,
        "wbc": 7
    }

    current = {
        "bp": df["bp"].iloc[-1],
        "heart_rate": df["heart_rate"].iloc[-1],
        "lactate": df["lactate"].iloc[-1],
        "wbc": df["wbc"].iloc[-1]
    }

    comp_df = pd.DataFrame({
        "Parameter": list(normal.keys()),
        "Patient": list(current.values()),
        "Normal": list(normal.values())
    })

    st.bar_chart(comp_df.set_index("Parameter"))        
            
 # ---------------- SUMMARY ---------------- #
        st.markdown("<div class='glass'><h3>📋 Clinical Summary</h3></div>", unsafe_allow_html=True)

        summary = {
            "BP": df["bp"].iloc[-1],
            "Lactate": df["lactate"].iloc[-1],
            "Heart Rate": df["heart_rate"].iloc[-1],
            "WBC": df["wbc"].iloc[-1]
        }

        st.table(pd.DataFrame(summary.items(), columns=["Parameter","Value"]))

        # ---------------- GAUGE ---------------- #
        st.markdown("## 🩺 Risk Meter")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Septic Shock Risk"},
            gauge={
                'axis': {'range': [0,1]},
                'steps': [
                    {'range': [0,0.4], 'color': "green"},
                    {'range': [0.4,0.7], 'color': "orange"},
                    {'range': [0.7,1], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        # ---------------- TRENDS ---------------- #
        st.markdown("## 📈 Vital Trends")

        df_plot = pd.DataFrame(data, columns=FEATURE_NAMES)

        fig = px.line(df_plot,
                      y=["bp","heart_rate","lactate","wbc"],
                      markers=True)

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- RISK TREND ---------------- #
        st.markdown("## 📊 Risk Progression")

        risk_curve = np.linspace(0, pred, 24)

        fig2 = px.line(y=risk_curve, title="Risk Growth Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        # ---------------- INSIGHTS ---------------- #
        st.markdown("<div class='glass'><h3>🧠 Medical Insights</h3></div>", unsafe_allow_html=True)

        insights = []
        precautions = []

        if df["lactate"].iloc[-1] > 2.5:
            insights.append("High lactate → tissue hypoxia")

        if df["bp"].iloc[-1] < 90:
            insights.append("Low BP → shock condition")

        if df["heart_rate"].iloc[-1] > 110:
            insights.append("High HR → stress response")

        if df["wbc"].iloc[-1] > 12:
            insights.append("High WBC → infection")

        precautions = [
            "Start IV fluids",
            "Administer vasopressors",
            "Monitor heart",
            "Start antibiotics"
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Conditions")
            for i in insights:
                st.markdown(f"- {i}")

        with col2:
            st.markdown("### 🛡️ Precautions")
            for p in precautions:
                st.markdown(f"- {p}")

        # ---------------- DOWNLOAD REPORT ---------------- #
        report = f"""
Septic Shock Report

Risk Score: {pred:.2f}
Status: {status}

Insights:
{insights}

Precautions:
{precautions}
"""

        st.download_button("📄 Download Report", report, "report.txt")

        # ---------------- FINAL ---------------- #
        st.markdown("<div class='glass'><h3>📌 Final Diagnosis</h3></div>", unsafe_allow_html=True)

        if pred > 0.7:
            st.error("⚠️ Immediate ICU intervention required")
        elif pred > 0.4:
            st.warning("🟠 Monitor closely")
        else:
            st.success("✅ Stable condition")
