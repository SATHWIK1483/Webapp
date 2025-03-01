import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Loading the saved model
loaded_model = pickle.load(open('final_model (2).sav', 'rb'))

# Creating a function for Prediction
@st.cache_data
def predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType):
    input_data = np.array([[card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType]])
    prediction = loaded_model.predict_proba(input_data)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    html_temp = """
        <div style="background-color:#000000 ;padding:10px">
        <h1 style="color:white;text-align:center;">Financial Transaction Fraud Detection System 💰 </h1>
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Load and display the banner image
    image = Image.open('home_banner.PNG')
    st.image(image, caption='Enhancing Financial Security with AI-powered Fraud Detection')

    # Sidebar Inputs
    st.sidebar.title("Fraud Risk Analysis System 🕵")
    st.sidebar.subheader("Enter Transaction Details for Risk Assessment")

    TransactionAmt = st.sidebar.number_input("Transaction Amount (in USD)", min_value=0, max_value=20000, step=1)
    card1 = st.sidebar.number_input("Primary Card Number (anonymized)", min_value=0, max_value=20000, step=1)
    card2 = st.sidebar.number_input("Secondary Card Number (anonymized)", min_value=0, max_value=20000, step=1)
    card4 = st.sidebar.radio("Card Issuer (Visa, Mastercard, etc.)", [1, 2, 3, 4])
    card6 = st.sidebar.radio("Card Type (Credit/Debit)", [1, 2])
    addr1 = st.sidebar.slider("Billing ZIP Code", min_value=0, max_value=500, step=1)
    addr2 = st.sidebar.slider("Billing Country Code", min_value=0, max_value=100, step=1)
    P_emaildomain = st.sidebar.selectbox("Purchaser Email Domain Category", [0, 1, 2, 3, 4])
    ProductCD = st.sidebar.selectbox("Product Category Code", [0, 1, 2, 3, 4])
    DeviceType = st.sidebar.radio("Device Used for Transaction (Mobile/Desktop)", [1, 2])

    # Display Result
    if st.button("Analyze Transaction for Fraud Risk"):
        output = predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType)
        final_output = output * 100
        st.subheader(f'Fraud Probability Score: {final_output}%')

        # Keeping result logic the same but modifying fraud/legitimate messages based on ProductCD
        if final_output > 75.0:
            if ProductCD % 2 == 0:
                st.error("🚨 ALERT! High-risk Fraud Detected 🚨")
            else:
                st.error("⚠ Warning: This Transaction is Likely Fraudulent! ⚠")
        else:
            if ProductCD % 2 == 0:
                st.success("✅ Secure Transaction: No Fraud Detected ✅")
            else:
                st.success("🎉 Transaction Verified as Legitimate 🎉")

if __name__ == "__main__":
    main()
