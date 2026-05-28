import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_model():
    df = pd.read_csv(r'C:\Users\vicen\Desktop\fraud-detection\creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, df

st.title("💳 Credit Card Fraud Detector")
st.markdown("Load a random transaction and find out if it's fraudulent or legitimate.")

with st.spinner("Loading model..."):
    model, scaler, df = load_model()

st.success("Model loaded successfully ✅")
st.warning("⚠️ This model detects 77% of fraudulent transactions. All flagged transactions should be reviewed by a human analyst before taking action.")

features = df.drop('Class', axis=1).columns.tolist()

# ── RANDOM TRANSACTION ────────────────────────────────────────
if st.button("🎲 Load Random Transaction"):
    random_row = df.sample(1).iloc[0]
    st.session_state.example = random_row.drop('Class')
    st.session_state.real_label = int(random_row['Class'])

if 'example' in st.session_state:
    example = st.session_state.example

    st.markdown("### Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Amount", f"€{example['Amount']:.2f}")
    with col2:
        st.metric("Time", f"{example['Time']:.0f}s")
    with col3:
        st.metric("V1", f"{example['V1']:.4f}")

    if st.button("🔍 Analyze Transaction"):
        input_df = pd.DataFrame([example])[features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.error(f"⚠️ FRAUDULENT — Confidence: {probability[1]*100:.1f}%")
        else:
            st.success(f"✅ LEGITIMATE — Confidence: {probability[0]*100:.1f}%")

        st.markdown("---")
        st.markdown("### 🔓 Real Answer")
        real = st.session_state.real_label
        if real == 1:
            st.error("This transaction was actually FRAUDULENT")
        else:
            st.success("This transaction was actually LEGITIMATE")

        if prediction == real:
            st.info("✓ Model was CORRECT")
        else:
            st.warning("✗ Model was WRONG")