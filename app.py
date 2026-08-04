import streamlit as st
import joblib

model = joblib.load("final_model.pkl")

st.write("Hello, World!")