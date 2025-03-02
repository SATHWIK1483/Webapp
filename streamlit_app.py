import numpy as np
import pickle
import time
import streamlit as st
from PIL import Image

# Load the saved model
@st.cache_resource
def load_model():
    return pickle.load(open('final_model.sav', 'rb'))  # Ensure correct file name

loaded_model = load_model()

# Streamlit app title and banner
html_temp = """
    <div style="background-color:#000000 ;padding:10px">
    <h1 style="color:white;text-align:center;">Financial Transaction Fraud Prediction ML Web App 💰 </h1>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Load and display the banner image
image = Image.open('home_banner.PNG')
st.image(image, caption='Impacting the World of Finance and Banking with Artificial Intelligence (AI)')

# Sidebar input options
st.sidebar.title("Financial Transaction Fraud Prediction System 🕵️")
st.sidebar.subheader("Choose the Below Parameters to Predict a Financial Transaction")

# Collect user inputs
TransactionAmt = st.sidebar.number_input("Choose the Transaction Amount in USD", 0, 20000, step=1)
card1 = st.sidebar.number_input("Choose the Payment Card 1 Amount (USD)", 0, 20000, step=1)
card2 = st.sidebar.number_input("Choose the Payment Card 2 Amount (USD)", 0, 20000, step=1)
card4 = st.sidebar.radio("Choose the Payment Card Category", [1, 2, 3, 4])
card6 = st.sidebar.radio("Choose the Payment Card Type", [1, 2])
addr1 = st.sidebar.slider("Choose the Payment Billing Zip Code", 0, 500, step=1)
addr2 = st.sidebar.slider("Choose the Payment Billing Country Code", 0, 100, step=1)
P_emaildomain = st.sidebar.selectbox("Choose the Purchaser Email Domain", [0, 1, 2, 3, 4])
ProductCD = st.sidebar.selectbox("Choose the Product Code", [0, 1, 2, 3, 4])
DeviceType = st.sidebar.radio("Choose the Payment Device Type", [1, 2])

# Fraud Prediction Function
def predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType):
    input_data = np.array([[card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType]])
    prediction = loaded_model.predict_proba(input_data)
    return float(prediction[0][1]) * 100  # Assuming fraud probability is at index 1

# Fraud Detection Images
safe_html = """ 
    <img src="https://media.giphy.com/media/g9582DNuQppxC/giphy.gif" alt="confirmed" style="width:698px;height:350px;"> 
"""
danger_html = """  
    <img src="https://media.giphy.com/media/8ymvg6pl1Lzy0/giphy.gif" alt="cancel" style="width:698px;height:350px;">
"""

# Predict Button
if st.button("Click Here To Predict"):
    final_output = predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType)
    st.subheader(f'Probability Score of Financial Transaction is {final_output:.2f}%')

    if final_output > 75.0:
        st.markdown(danger_html, unsafe_allow_html=True)
        st.error("**OMG! Financial Transaction is Fraud**")
    else:
        st.balloons()
        st.markdown(safe_html, unsafe_allow_html=True)
        st.success("**Hurray! Transaction is Legitimate**")

# Define the main function
def main():
    pass  # The script should work without an explicit main function

# Run the App
if __name__ == "__main__":
    main()
