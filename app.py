import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
# Make sure 'model.pkl' is in the same directory as this script.
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

def predict_loan_status(features):
    """
    Uses the loaded model to make a prediction.
    """
    # The model expects a 2D array, so we reshape it
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

    st.title("🏦 Loan Approval Prediction")
    st.write("Enter the applicant's details below to predict the loan approval status.")

    # --- Create input fields for user data ---
    # The order of these columns must match the order of features the model was trained on.
    # Based on the notebook: ['Gender', 'Married', 'Education', 'Self_Employed',
    # 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    # 'Credit_History', 'Property_Area']

    # Mapping dictionaries for categorical features
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 0, 'Not Graduate': 1}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    credit_history_map = {'Yes (Debt paid)': 1.0, 'No (Debt not paid)': 0.0}

    # Creating columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', list(gender_map.keys()))
        education = st.selectbox('Education', list(education_map.keys()))
        applicant_income = st.number_input('Applicant Income ($)', min_value=0, value=5000)
        loan_amount = st.number_input('Loan Amount ($)', min_value=0, value=150)
        credit_history = st.selectbox('Credit History Available?', list(credit_history_map.keys()))

    with col2:
        married = st.selectbox('Married', list(married_map.keys()))
        self_employed = st.selectbox('Self Employed', list(self_employed_map.keys()))
        coapplicant_income = st.number_input('Co-applicant Income ($)', min_value=0, value=1600)
        loan_amount_term = st.number_input('Loan Amount Term (Months)', min_value=12, value=360, step=12)
        property_area = st.selectbox('Property Area', list(property_area_map.keys()))

    # --- Preprocess inputs and make prediction ---
    if st.button('Predict Loan Status', type="primary"):
        # Convert user inputs to the numerical format the model expects
        gender_num = gender_map[gender]
        married_num = married_map[married]
        education_num = education_map[education]
        self_employed_num = self_employed_map[self_employed]
        property_area_num = property_area_map[property_area]
        credit_history_num = credit_history_map[credit_history]

        # Assemble the feature vector in the correct order
        features = [
            gender_num, married_num, education_num, self_employed_num,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history_num, property_area_num
        ]

        # Make a prediction
        prediction = predict_loan_status(features)

        # Display the result
        st.write("---")
        if prediction[0] == 1:
            st.success('🎉 Congratulations! The loan is likely to be **Approved**.')
        else:
            st.error('😞 Unfortunately, the loan is likely to be **Rejected**.')

if __name__ == '__main__':
    main()