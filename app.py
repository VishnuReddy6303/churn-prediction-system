import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("xgb_churn_model.pkl")

st.title("Churn Prediction System 🚀")

st.write("Enter customer details to predict churn:")

# Example input fields (adjust according to your dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)

# Create DataFrame for input
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges]
})

# If your model needs encoding for categorical features:
input_df = pd.get_dummies(input_df)  # simple one-hot encoding

# Match columns with training features
# Ensure all training columns exist in input
missing_cols = set(model.get_booster().feature_names) - set(input_df.columns)
for c in missing_cols:
    input_df[c] = 0  # fill missing columns with 0

input_df = input_df[model.get_booster().feature_names]  # reorder columns

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[:,1]
    if prediction[0] == 1:
        st.error(f"Customer is likely to churn. Probability: {proba[0]*100:.2f}%")
    else:
        st.success(f"Customer is likely to stay. Probability: {proba[0]*100:.2f}%")
