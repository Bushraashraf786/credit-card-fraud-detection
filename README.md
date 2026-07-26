# 💳 Credit Card Fraud Detection

A machine learning web app that predicts whether a credit card transaction is legitimate or fraudulent, built using Logistic Regression and deployed with Streamlit.

## 📌 Overview

This project uses a Logistic Regression model trained on transaction data (Time, Amount, and PCA-transformed features V1–V28) to classify transactions as **Legit** or **Fraud**. Since fraud cases are rare in real-world data, the dataset was balanced through undersampling before training.

## 🎯 Model Performance

- **Algorithm:** Logistic Regression
- **Accuracy:** ~85%

## 🛠️ Tech Stack

- Python
- Scikit-learn (model training)
- Pandas & NumPy (data processing)
- Streamlit (web app deployment)

## 📂 Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `credit_card_model.pkl` | Trained Logistic Regression model |
| `requirements.txt` | Python dependencies |

## 🚀 How It Works

1. User enters transaction details (Time, Amount, PCA features V1–V8).
2. The app feeds this data into the trained model.
3. The model predicts whether the transaction is **Legitimate ✅** or **Fraudulent 🚨**.

## 📊 Dataset

Trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains anonymized transactions made by European cardholders.

## 👤 Author

Bushra Ashraf
