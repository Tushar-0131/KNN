import streamlit as st
import pickle
import numpy as np

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

st.title("Health Condition Prediction App")
st.write("Enter the following details to get a prediction:")

# Create input fields based on model features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    bp_sys = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    bp_dia = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
    smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    alcohol = st.selectbox("Alcohol Intake", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    activity = st.selectbox("Physical Activity", options=[0, 1], format_func=lambda x: "Active" if x == 1 else "Inactive")
    family_hist = st.selectbox("Family History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

# Prepare input for prediction
features = np.array([[
    age, bmi, bp_sys, bp_dia, cholesterol, glucose, 
    smoking, alcohol, activity, family_hist, heart_disease, 
    diabetes, gender
]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.subheader(f"Prediction Result: {prediction[0]}")
