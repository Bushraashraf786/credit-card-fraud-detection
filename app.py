import streamlit as st
import numpy as np
import pickle

# Model Load Karein
model = pickle.load(open('credit_card_model.pkl', 'rb'))

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Transaction ki details bhar kar check karein ke Legit hai ya Fraud.")

with st.expander("ℹ️ What do these fields mean and how to fill them? (Click here)"):
    st.markdown("""
    - **Transaction Time (Seconds):** The time elapsed since the first transaction in the dataset (higher = later transaction).
    - **Transaction Amount ($):** The dollar amount of the transaction. Larger amounts tend to raise fraud suspicion.
    - **V1 to V8 (PCA Features):** These are NOT real customer details (name, location, etc.) — for privacy reasons,
      the original data was mathematically transformed into these anonymized numbers. They don't have a direct meaning.
      - **0.00** = a normal/average value
      - **Negative value** (e.g. -5) = an "unusual" pattern
      - **Positive value** (e.g. +5) = a different pattern, but there's no fixed rule linking sign to fraud/legit
    - **V9 to V28:** Not shown in this app — automatically set to 0.0 (average) in the background.
    - **Predict Button:** Sends all the values to the model and returns whether the transaction is Legit or Fraud.
    """)

time = st.number_input("Transaction Time (Seconds)", value=0.0)
amount = st.number_input("Transaction Amount ($)", value=0.0)

st.subheader("PCA Features (V1 to V8)")
st.caption("Zyada tar cases mein 0.00 hi rehne dein — test karne k liye chahein to badal sakte hain.")
col1, col2 = st.columns(2)
with col1:
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
with col2:
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    v7 = st.number_input("V7", value=0.0)
    v8 = st.number_input("V8", value=0.0)

# Baqi V9-V28 features default 0.0
remaining_v = [0.0] * 20

if st.button("Predict Transaction Status", type="primary"):
    input_data = [time, v1, v2, v3, v4, v5, v6, v7, v8] + remaining_v + [amount]
    features = np.array([input_data])
    prediction = model.predict(features)

    st.markdown("---")
    if prediction == 0:
        st.success("### ✅ Ye LEGITIMATE Transaction hai.")
    else:
        st.error("### 🚨 Ye FRAUDULENT Transaction hai!")
