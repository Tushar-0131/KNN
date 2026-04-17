import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Prediction Tool", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header Section
st.title("🩺 Health Condition Predictor")
st.markdown("""
    This app uses a **Machine Learning model** (K-Neighbors Classifier) to predict health outcomes 
    based on clinical and lifestyle data. Please fill in the details below.
""")

st.divider()

# Input Section: Using columns for a better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Metrics")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)
    bp_sys = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
    bp_dia = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=190)
    gluc = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)

with col2:
    st.subheader("Lifestyle & History")
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Intake?", ["No", "Yes"])
    activity = st.selectbox("Physically Active?", ["No", "Yes"])
    fam_hist = st.selectbox("Family History of Disease?", ["No", "Yes"])
    heart_dis = st.selectbox("Existing Heart Disease?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes Diagnosis?", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

# Map categorical inputs back to 0 and 1 for the model
binary_map = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}

# Prepare the data for prediction
# Note: The order follows the internal structure of your model.pkl
input_data = pd.DataFrame([[
    age, bmi, bp_sys, bp_dia, chol, gluc,
    binary_map[smoking], binary_map[alcohol], binary_map[activity],
    binary_map[fam_hist], binary_map[heart_dis], binary_map[diabetes],
    gender_map[gender]
]], columns=[
    'Age', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
    'Cholesterol', 'Glucose_Level', 'Smoking', 'Alcohol_Intake', 
    'Physical_Activity', 'Family_History', 'Heart_Disease', 'Diabetes', 'Gender_Male'
])

st.divider()

# Prediction Button
if st.button("Generate Prediction Result"):
    # Perform prediction
    prediction = model.predict(input_data)
    
    # Visual Output
    st.subheader("Analysis Results:")
    
    # Customizing the output based on common classification labels (e.g., 0 for Healthy, 1 for At Risk)
    if prediction[0] == 0:
        st.success("✅ **Result: Low Risk / Normal**")
        st.write("Based on the provided metrics, the model suggests that the health indicators are within a normal range.")
    else:
        st.error("⚠️ **Result: High Risk / Condition Detected**")
        st.write("Based on the provided metrics, the model indicates a potential health risk. It is recommended to consult with a medical professional.")

    # Show prediction probability if the model supports it
    try:
        prob = model.predict_proba(input_data)
        confidence = np.max(prob) * 100
        st.info(f"Model Confidence: {confidence:.2f}%")
    except:
        pass

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")
