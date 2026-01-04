import streamlit as st
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Nairobi Housing Price Predictor",
    layout="centered"
)

st.title("🏠 Nairobi Housing Price Prediction")
st.write(
    "This application predicts residential housing prices in Nairobi "
    "using a machine learning regression model trained on real estate data."
)