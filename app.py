import streamlit as st
import pandas as pd
import joblib
from utils.preprocessing import preprocess_input  # your custom preprocessing
import gdown
import zipfile
import os
import joblib

@st.cache_data
def preprocess_cached(df):
    return preprocess_input(df)

@st.cache_resource
def load_model():
    model_path = 'model/churn_model.pkl'

    # Check if model already exists
    if not os.path.exists(model_path):
        # Download zip file
        url = 'https://drive.google.com/uc?id=1xOjyJ1BHOzOv5--fYuS1zAN_ebIfbKCa'  # Replace with your ID
        output = 'churn_model.zip'
        gdown.download(url, output, quiet=False)

        # Extract it
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()  # Extracts into working dir

    # Load the model
    return joblib.load(model_path)


# Load once, cached
model = load_model()
st.title("Churn Prediction App")
st.write("Please fill in the customer details:")

with st.form("input_form"):
    tenure = st.selectbox("TENURE", [
        "K > 24 month", "I 18-21 month", "G 12-15 month", "H 15-18 month",
        "J 21-24 month", "F 9-12 month", "D 3-6 month", "E 6-9 month"
    ])

    montant = st.number_input("MONTANT", min_value=0.0, step=0.01)
    frequence_rech = st.number_input("FREQUENCE_RECH", min_value=0, step=1)
    revenue = st.number_input("REVENUE", min_value=0.0, step=0.01)
    arpu_segment = st.selectbox("ARPU_SEGMENT", ["Low", "Medium", "High"])
    frequence = st.number_input("FREQUENCE", min_value=0, step=1)
    data_volume = st.number_input("DATA_VOLUME", min_value=0.0, step=0.01)
    on_net = st.number_input("ON_NET", min_value=0, step=1)
    orange = st.number_input("ORANGE", min_value=0, step=1)
    tigo = st.number_input("TIGO", min_value=0, step=1)
    regularity = st.number_input("REGULARITY", min_value=0, step=1)
    freq_top_pack = st.number_input("FREQ_TOP_PACK", min_value=0, step=1)
    region = st.selectbox("REGION", ["CENTRE", "EAST", "NORTH", "SOUTH", "WEST", "UNKNOWN"])
    top_pack = st.selectbox("TOP_PACK", ["Internet", "Orange", "Voice", "UNKNOWN"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame({
        'TENURE': [tenure],
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [data_volume],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'REGULARITY': [regularity],
        'FREQ_TOP_PACK': [freq_top_pack],
        'REGION': [region],
        'TOP_PACK': [top_pack],
    })

    # Preprocess input (handle categorical -> numeric conversion etc.)
    processed_input = preprocess_cached(input_df)

    # Predict churn
    prediction = model.predict(processed_input)[0]
    proba = model.predict_proba(processed_input)[0][1]  # probability of churn (class 1)

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn with a probability of {proba:.2%}.")
    else:
        st.success(f"✅ Customer is NOT likely to churn with a probability of {(1 - proba):.2%}.")
