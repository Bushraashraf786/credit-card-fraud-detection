import streamlit as st
import numpy as np
import pickle

# Model Load Karein
model = pickle.load(open('credit_card_model.pkl', 'rb'))

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Transaction ki details bhar kar check karein ke Legit hai ya Fraud.")

with st.expander("ℹ️ Ye fields kya hain aur values kaise dalein? (Click karein)"):
    st.markdown("""
    - **Transaction Time (Seconds):** Transaction ke waqt ka number (jitna zyada, utna baad ka transaction).
    - **Transaction Amount ($):** Kitne dollars ki transaction hui. Bara amount fraud ka shak barha deta hai.
    - **V1 se V8 (PCA Features):** Ye asli customer details (naam, location) nahi hain — privacy ki wajah se
      mathematical numbers mein convert ki gayi hain. Inka koi seedha matlab nahi hota.
      - **0.00** rakhna = normal/average value
      - **Negative value** (jaise -5) = kuch "unusual" pattern
      - **Positive value** (jaise +5) = alag pattern, lekin fraud/legit ka fixed rule nahi
    - **V9 se V28:** App mein nahi dikhaye gaye — automatically 0.0 (average) le liye jate hain.
    - **Predict Button:** Sab values ko model mein bhej kar Legit ya Fraud ka result deta hai.
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
