import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import streamlit as st
from src.utils.data_utils import prepare_input_data, validate_numeric_range
from src.utils.model_utils import load_model_artifacts, train_and_save_model
from src.utils.preprocessing_utils import (
    create_interaction_features,
    create_derived_features,
    encode_categorical_features
)
from src.utils.visualization_utils import (
    plot_prediction_gauge,
    create_prediction_summary
)

def create_input_form():
    """Create the input form for prediction"""
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input('Credit Score', 300, 900, 650)
            age = st.number_input('Age', 18, 100, 35)
            tenure = st.number_input('Tenure (years)', 0, 50, 5)
            balance = st.number_input('Balance', 0.0, 250000.0, 50000.0)
            products = st.number_input('Number of Products', 1, 4, 1)
        
        with col2:
            geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
            gender = st.selectbox('Gender', ['Male', 'Female'])
            has_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
            is_active = st.selectbox('Is Active Member', ['Yes', 'No'])
            salary = st.number_input('Estimated Salary', 0.0, 500000.0, 50000.0)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            input_data = {
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': products,
                'HasCrCard': 1 if has_card == 'Yes' else 0,
                'IsActiveMember': 1 if is_active == 'Yes' else 0,
                'EstimatedSalary': salary
            }
            return input_data
    return None

def encode_input_data(df, scaler, feature_names):
    """Encode and transform input data to match model requirements"""
    # Create dummy variables for Geography
    geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')
    
    # Create dummy variables for Gender
    gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
    
    # Drop original categorical columns
    df = df.drop(['Geography', 'Gender'], axis=1)
    
    # Concatenate dummy variables
    df = pd.concat([df, geography_dummies, gender_dummies], axis=1)
    
    # Ensure all feature columns from training are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    # Scale features
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_names)
    
    return df_scaled

def run_prediction_page():
    st.title("Customer Churn Prediction")
    
    # Load model artifacts
    try:
        model_dir = Path("models/trained")
        model, scaler, feature_names = load_model_artifacts(model_dir)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=5)
        
        with col2:
            balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=0.0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
            has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
            is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=500000.0, value=50000.0)
        
        submit_button = st.form_submit_button("Predict")
    
    if submit_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_credit_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active_member == "Yes" else 0],
            'EstimatedSalary': [estimated_salary]
        })
        
        # Create dummy variables
        input_encoded = pd.get_dummies(input_data, columns=['Geography', 'Gender'])
        
        # Ensure all necessary columns exist
        for feature in feature_names:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0
        
        # Select only the features used during training
        X = input_encoded[feature_names]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Display prediction
        st.header("Prediction Results")
        
        if prediction == 1:
            st.error(f"⚠️ High risk of churn! (Probability: {probability:.1%})")
        else:
            st.success(f"✅ Low risk of churn (Probability: {probability:.1%})")

if __name__ == "__main__":
    run_prediction_page() 