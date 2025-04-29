import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd

def plot_prediction_gauge(probability: float) -> None:
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    st.plotly_chart(fig)

def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Plot feature importance"""
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance'
    )
    st.plotly_chart(fig)

def create_prediction_summary(
    input_data: Dict,
    probability: float,
    top_features: List[str]
) -> None:
    """Create a summary of the prediction"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Summary")
        if probability > 0.5:
            st.error(f"High Risk of Churn: {probability:.1%}")
        else:
            st.success(f"Low Risk of Churn: {probability:.1%}")
    
    with col2:
        st.subheader("Top Contributing Factors")
        for feature in top_features:
            st.write(f"• {feature}") 