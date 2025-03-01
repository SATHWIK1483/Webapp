import numpy as np
import pickle
import streamlit as st
import shap
import lime
import lime.lime_tabular
import pandas as pd
from PIL import Image

# Load trained fraud detection model
loaded_model = pickle.load(open("final_model(2).sav", "rb"))

# Function for Fraud Prediction
@st.cache_data
def predict_fraud(features):
    prediction = loaded_model.predict_proba(features)
    return prediction[0][1] * 100  # Probability of fraud

def main():
    st.title("Financial Transaction Fraud Prediction System 💳")
    
    # Load and display the banner image
    image = Image.open('home_banner.PNG')
    st.image(image, caption='Impacting the World of Finance and Banking with AI')
    
    st.sidebar.title("Enter Transaction Details")
    
    # User Inputs
    TransactionAmt = st.sidebar.number_input("Transaction Amount (USD)", min_value=0.0, step=0.01)
    ProductCD = st.sidebar.selectbox("Product Code", ["A", "B", "C", "D", "E"])
    card1 = st.sidebar.number_input("Card ID (Hidden for security)", min_value=1000, max_value=999999, step=1)
    card2 = st.sidebar.number_input("Card 2 ID", min_value=100, max_value=9999, step=1)
    card4 = st.sidebar.radio("Payment Card Category", ["Visa", "Mastercard", "American Express", "Discover"])
    card6 = st.sidebar.radio("Payment Card Type", ["Credit", "Debit"])
    billing_zip = st.sidebar.slider("Billing Zip Code", min_value=0, max_value=9999)
    billing_country = st.sidebar.slider("Billing Country Code", min_value=0, max_value=100)
    p_emaildomain = st.sidebar.selectbox("Purchaser Email Domain", ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"])
    device_type = st.sidebar.radio("Device Type", ["Mobile", "Desktop"])
    
    # Convert Categorical Inputs to Model-Compatible Values
    card4_map = {"Visa": 1, "Mastercard": 2, "American Express": 3, "Discover": 4}
    card6_map = {"Credit": 1, "Debit": 2}
    device_map = {"Mobile": 1, "Desktop": 2}
    p_email_map = {"gmail.com": 1, "yahoo.com": 2, "hotmail.com": 3, "outlook.com": 4}
    product_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    
    # Create Feature Vector
    features = np.array([[
        TransactionAmt, card1, card2, card4_map[card4], card6_map[card6],
        billing_zip, billing_country, p_email_map[p_emaildomain], product_map[ProductCD], device_map[device_type]
    ]])
    
    # Make Prediction
    if st.button("Predict Fraudulent Transaction"):
        fraud_probability = predict_fraud(features)
        st.subheader(f'Probability of Fraud: {fraud_probability:.2f}%')
        
        if fraud_probability > 75.0:
            st.error("🚨 ALERT! High-risk Fraud Detected 🚨")
        else:
            st.success("✅ Safe Transaction: No Fraud Detected ✅")

        # Explainability using SHAP
        explainer = shap.TreeExplainer(loaded_model)
        shap_values = explainer.shap_values(pd.DataFrame(features, columns=["TransactionAmt", "card1", "card2", "card4", "card6", "addr1", "addr2", "P_emaildomain", "ProductCD", "DeviceType"]))
        st.subheader("Feature Contribution (SHAP)")
        shap.summary_plot(shap_values, pd.DataFrame(features, columns=["TransactionAmt", "card1", "card2", "card4", "card6", "addr1", "addr2", "P_emaildomain", "ProductCD", "DeviceType"]))

        # Explainability using LIME
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.random.rand(100, 10), mode="classification"
        )
        exp = lime_explainer.explain_instance(features[0], loaded_model.predict_proba)
        st.subheader("Feature Importance (LIME)")
        st.write(exp.as_list())

if _name_ == "_main_":
    main()

