import numpy as np
import pickle
import time
import streamlit as st
from PIL import Image

# Load the saved model (update the filename as needed)
loaded_model = pickle.load(open('final_model (1).sav', 'rb'))  # Change if using 'final_model (2).sav'

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

# Transaction Amount
st.sidebar.markdown("### Transaction Amount")
TransactionAmt = st.sidebar.number_input("Choose the Transaction Amount in USD", 0, 20000, step=1)

# Payment Card 1
st.sidebar.markdown("### Payment Card 1")
card1 = st.sidebar.number_input("Choose the Payment Card 1 Amount (USD)", 0, 20000, step=1)

# Payment Card 2
st.sidebar.markdown("### Payment Card 2")
card2 = st.sidebar.number_input("Choose the Payment Card 2 Amount (USD)", 0, 20000, step=1)

# Payment Card Category
st.sidebar.markdown("### Payment Card Category")
card4 = st.sidebar.radio("Choose the Payment Card Category", [1, 2, 3, 4])
st.sidebar.info("1 : Discover | 2 : Mastercard | 3 : American Express | 4 : Visa")

# Payment Card Type
st.sidebar.markdown("### Payment Card Type")
card6 = st.sidebar.radio("Choose the Payment Card Type", [1, 2])
st.sidebar.info("1 : Credit | 2 : Debit")

# Billing Zip Code
st.sidebar.markdown("### Billing Zip Code")
addr1 = st.sidebar.slider("Choose the Payment Billing Zip Code", 0, 500, step=1)

# Billing Country Code
st.sidebar.markdown("### Billing Country Code")
addr2 = st.sidebar.slider("Choose the Payment Billing Country Code", 0, 100, step=1)

# Purchaser Email Domain
st.sidebar.markdown("### Purchaser Email Domain")
P_emaildomain = st.sidebar.selectbox("Choose the Purchaser Email Domain", [0, 1, 2, 3, 4])
st.sidebar.info("0 : Gmail (Google) | 1 : Outlook (Microsoft) | 2 : Mail.com | 3 : Others | 4 : Yahoo")

# Product Code
st.sidebar.markdown("### Product Code")
ProductCD = st.sidebar.selectbox("Choose the Product Code", [0, 1, 2, 3, 4])
st.sidebar.info("0 : C | 1 : H | 2 : R | 3 : S | 4 : W")

# Device Type
st.sidebar.markdown("### Device Type")
DeviceType = st.sidebar.radio("Choose the Payment Device Type", [1, 2])
st.sidebar.info("1 : Mobile | 2 : Desktop")

# Fraud Prediction Function
@st.cache_data
def predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType):
    input_data = np.array([[card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType]])
    prediction = loaded_model.predict_proba(input_data)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

# Fraud Detection Images
safe_html = """ 
    <img src="https://media.giphy.com/media/g9582DNuQppxC/giphy.gif" alt="confirmed" style="width:698px;height:350px;"> 
"""
danger_html = """  
    <img src="https://media.giphy.com/media/8ymvg6pl1Lzy0/giphy.gif" alt="cancel" style="width:698px;height:350px;">
"""

# Predict Button
if st.button("Click Here To Predict"):
    output = predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType)
    final_output = output * 100
    st.subheader('Probability Score of Financial Transaction is {}% '.format(final_output))

    if final_output > 75.0:
        st.markdown(danger_html, unsafe_allow_html=True)
        st.error("**OMG! Financial Transaction is Fraud**")
    else:
        st.balloons()
        time.sleep(5)
        st.balloons()
        time.sleep(5)
        st.balloons()
        st.markdown(safe_html, unsafe_allow_html=True)
        st.success("**Hurray! Transaction is Legitimate**")

# Run the App
if __name__ == '__main__':
    main()
