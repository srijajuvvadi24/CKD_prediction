import streamlit as st
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CKD Prediction System",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ============================= */
/* OVERALL WEBSITE ZOOM */
/* ============================= */

html, body, [class*="css"]  {
    font-size: 18px;
}

/* ============================= */
/* MAIN CONTAINER */
/* ============================= */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1500px;
}

/* ============================= */
/* TITLES */
/* ============================= */

h1 {
    font-size: 52px !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 34px !important;
}

h3 {
    font-size: 28px !important;
}

/* ============================= */
/* TEXT */
/* ============================= */

p {
    font-size: 18px !important;
}

/* ============================= */
/* RADIO BUTTONS */
/* ============================= */

div[role="radiogroup"] label {
    font-size: 18px !important;
    font-weight: 600;
}

/* ============================= */
/* INPUT LABELS */
/* ============================= */

label {
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* ============================= */
/* INPUT BOXES */
/* ============================= */

div[data-baseweb="input"] input {
    font-size: 18px !important;
    height: 50px;
}

/* ============================= */
/* METRIC CARDS */
/* ============================= */

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}

/* Metric Values */

div[data-testid="stMetricValue"] {
    font-size: 34px !important;
    font-weight: bold;
}

/* Metric Labels */

div[data-testid="stMetricLabel"] {
    font-size: 18px !important;
}

/* ============================= */
/* BUTTON */
/* ============================= */

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    font-size: 20px;
    font-weight: bold;
}

/* ============================= */
/* ALERT BOXES */
/* ============================= */

div[data-testid="stAlert"] {
    font-size: 18px;
    border-radius: 12px;
}

/* ============================= */
/* FOOTER */
/* ============================= */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================

rf_data = pickle.load(open("ckd_model.pkl", "rb"))

rf_model = rf_data["base_model"]

ann_model = load_model("ann_model.h5")

scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================================================
# TITLE
# =========================================================

st.title("Chronic Kidney Disease Prediction System")

st.write(
    "AI-based system for predicting Chronic Kidney Disease "
    "using Machine Learning and Deep Learning algorithms."
)

st.markdown("---")

# =========================================================
# MODEL SELECTION
# =========================================================

st.subheader("Select Prediction Model")

model_choice = st.radio(
    "Choose Model",
    ["Random Forest", "ANN"],
    horizontal=True
)

st.markdown("---")

# =========================================================
# MAIN LAYOUT
# =========================================================

left_col, right_col = st.columns([0.9, 1.1])

# =========================================================
# LEFT SIDE → INPUT SECTION
# =========================================================

with left_col:

    st.subheader("Enter Patient Clinical Details")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=45,
        step=1
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=50,
        max_value=200,
        value=80,
        step=1
    )

    sg = st.number_input(
        "Specific Gravity",
        min_value=1.000,
        max_value=1.030,
        value=1.020,
        step=0.001,
        format="%.3f"
    )

    al = st.number_input(
        "Albumin",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )

    su = st.number_input(
        "Sugar",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )

    hemo = st.number_input(
        "Hemoglobin",
        min_value=3.0,
        max_value=18.0,
        value=13.5,
        step=0.1
    )

    predict_btn = st.button("Predict CKD")

# =========================================================
# RIGHT SIDE → RESULTS SECTION
# =========================================================

with right_col:

    st.subheader("Model Performance")

    # ---------------- RANDOM FOREST ----------------

    if model_choice == "Random Forest":

        st.success("Random Forest Model")

        col1, col2 = st.columns(2)

        col1.metric("Accuracy", "94%")
        col1.metric("Precision", "95%")

        col2.metric("Recall", "97%")
        col2.metric("F1-Score", "96%")

    # ---------------- ANN ----------------

    else:

        st.success("ANN Model")

        col1, col2 = st.columns(2)

        col1.metric("Accuracy", "92%")
        col1.metric("Precision", "93%")

        col2.metric("Recall", "95%")
        col2.metric("F1-Score", "94%")

    st.markdown("---")

    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_btn:

        input_data = pd.DataFrame([[
            age,
            bp,
            sg,
            al,
            su,
            hemo
        ]], columns=[
            'age',
            'bp',
            'sg',
            'al',
            'su',
            'hemo'
        ])

        # ---------------- RANDOM FOREST ----------------

        if model_choice == "Random Forest":

            prediction = rf_model.predict(input_data)[0]

            probability = rf_model.predict_proba(input_data)[0][1]

        # ---------------- ANN ----------------

        else:

            scaled_data = scaler.transform(input_data)

            probability = ann_model.predict(scaled_data)[0][0]

            prediction = 1 if probability > 0.5 else 0

        # =================================================
        # RESULT SECTION
        # =================================================

        st.markdown("## Prediction Result")

        # ---------------- CKD ----------------

        if prediction == 1:

            st.markdown("""
            <div style="
            padding:20px;
            border-radius:12px;
            background-color:#3b0a0a;
            color:white;
            font-size:22px;
            font-weight:bold;
            text-align:center;">
            ⚠ Patient is likely to have CKD
            </div>
            """, unsafe_allow_html=True)

        # ---------------- NO CKD ----------------

        else:

            st.markdown("""
            <div style="
            padding:20px;
            border-radius:12px;
            background-color:#052e16;
            color:white;
            font-size:22px;
            font-weight:bold;
            text-align:center;">
            ✔ Patient is unlikely to have CKD
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # PROBABILITY
        # =================================================

        risk_percent = round(probability * 100, 2)

        st.metric(
            "Prediction Probability",
            f"{risk_percent}%"
        )

        # =================================================
        # RISK LEVEL
        # =================================================

        if risk_percent < 40:

            st.info("Risk Level : LOW")

        elif risk_percent < 70:

            st.warning("Risk Level : MODERATE")

        else:

            st.error("Risk Level : HIGH")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "CKD Prediction using Random Forest and Artificial Neural Network"
)
