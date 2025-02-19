import numpy as np
import pickle
import time
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
        <h1 style="color:white;text-align:center;">Financial Transaction Fraud Prediction ML Web App 💰 </h1>
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Load and display the banner image
    image = Image.open('home_banner.PNG')
    st.image(image, caption='Impacting the World of Finance and Banking with Artificial Intelligence (AI)')

    # Sidebar Inputs
    st.sidebar.title("Financial Transaction Fraud Prediction System 🕵️")
    st.sidebar.subheader("Choose the Below Parameters to Predict a Financial Transaction")

    TransactionAmt = st.sidebar.number_input("Transaction Amount in USD", 0, 20000, step=1)
    card1 = st.sidebar.number_input("Payment Card 1 Amount (USD)", 0, 20000, step=1)
    card2 = st.sidebar.number_input("Payment Card 2 Amount (USD)", 0, 20000, step=1)
    card4 = st.sidebar.radio("Payment Card Category", [1, 2, 3, 4])
    card6 = st.sidebar.radio("Payment Card Type", [1, 2])
    addr1 = st.sidebar.slider("Billing Zip Code", 0, 500, step=1)
    addr2 = st.sidebar.slider("Billing Country Code", 0, 100, step=1)
    P_emaildomain = st.sidebar.selectbox("Purchaser Email Domain", [0, 1, 2, 3, 4])
    ProductCD = st.sidebar.selectbox("Product Code", [0, 1, 2, 3, 4])
    DeviceType = st.sidebar.radio("Device Type", [1, 2])

    # Display Result
    if st.button("Click Here To Predict"):
        output = predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType)
        final_output = output * 100
        st.subheader(f'Probability Score of Financial Transaction is {final_output}%')

        if final_output > 75.0:
            st.error("**OMG! Financial Transaction is Fraud**")
        else:
            st.success("**Hurray! Transaction is Legitimate**")

if __name__ == '__main__':
    main()
