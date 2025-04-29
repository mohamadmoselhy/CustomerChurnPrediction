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
    st.title('Customer Churn Prediction')
    
    # Check if model exists, if not, train and save it
    model_dir = Path('models/trained')
    if not model_dir.exists() or not (model_dir / 'model_ensemble.pkl').exists():
        st.warning("Model not found. Training new model...")
        try:
            train_and_save_model()
            st.success("Model trained and saved successfully!")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return
    
    try:
        model, scaler, feature_names = load_model_artifacts('models/trained')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    st.write('Enter customer information to predict churn probability')
    
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
            try:
                # Create input DataFrame
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
                
                # Create DataFrame
                df = pd.DataFrame([input_data])
                
                # Encode and transform input data
                df_processed = encode_input_data(df, scaler, feature_names)
                
                # Make prediction
                probability = model.predict_proba(df_processed)[0, 1]
                
                # Display results
                st.subheader('Prediction Results')
                
                # Create color-coded probability gauge
                prob_color = 'red' if probability > 0.5 else 'green'
                st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="color: {prob_color};">
                        Churn Probability: {probability:.1%}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display interpretation
                if probability > 0.5:
                    st.error('⚠️ High Risk of Churn')
                    st.write('This customer is likely to churn. Consider immediate retention actions.')
                    
                    # Add recommendations for high-risk customers
                    st.subheader("Recommended Actions:")
                    st.write("1. Contact customer for feedback")
                    st.write("2. Offer personalized retention package")
                    st.write("3. Review pricing and product fit")
                else:
                    st.success('✅ Low Risk of Churn')
                    st.write('This customer is likely to stay. Consider growth opportunities.')
                    
                    # Add recommendations for low-risk customers
                    st.subheader("Growth Opportunities:")
                    st.write("1. Cross-selling opportunities")
                    st.write("2. Loyalty program enrollment")
                    st.write("3. Service upgrades")
                
                # Display feature importance
                st.subheader("Key Factors Influencing Prediction")
                feature_importance = pd.DataFrame({
                    'Feature': df_processed.columns,
                    'Value': df_processed.iloc[0].values
                })
                st.dataframe(feature_importance.sort_values('Value', ascending=False))
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Debug information:")
                st.write(f"Feature names expected: {feature_names}")
                st.write(f"Input data columns: {df.columns.tolist()}") 