import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
from src.utils.data_utils import load_real_data
from src.utils.model_utils import evaluate_model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

def load_model_artifacts(model_dir):
    """Load model and related artifacts"""
    model_dir = Path(model_dir)
    
    with open(model_dir / 'model_ensemble.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(model_dir / 'model_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'model_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names

def create_sample_data(n_samples=1000):
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'CreditScore': np.random.randint(300, 900, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 100, n_samples),
        'Tenure': np.random.randint(0, 50, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.randint(1, 5, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(0, 500000, n_samples),
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    return data

def display_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    """Display model performance metrics and visualizations"""
    # Calculate metrics
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)
    
    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        st.pyplot(plt)

def run_dashboard_page():
    st.title("Customer Churn Dashboard")
    
    # Load the real data
    data = load_real_data()
    
    if data is None:
        st.warning("No data available. Please ensure the data file exists.")
        return
    
    # Load model artifacts
    try:
        model_dir = Path("models/trained")
        model, scaler, feature_names = load_model_artifacts(model_dir)
        
        # Prepare data for prediction
        # First, create dummy variables for categorical features
        data_encoded = pd.get_dummies(data, columns=['Geography', 'Gender'])
        
        # Ensure all necessary columns exist (add missing ones with zeros if needed)
        for feature in feature_names:
            if feature not in data_encoded.columns:
                data_encoded[feature] = 0
                
        # Select only the features used during training
        X = data_encoded[feature_names]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        y_true = data['Exited']
        
        # Get predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Handle missing values in Geography
    data['Geography'] = data['Geography'].fillna('Unknown')
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(data)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_rate = (data['Exited'].mean() * 100)
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_credit = data['CreditScore'].mean()
        st.metric("Avg Credit Score", f"{avg_credit:.0f}")
    
    with col4:
        avg_balance = data['Balance'].mean()
        st.metric("Avg Balance", f"${avg_balance:,.0f}")
    
    # Filters
    st.sidebar.header("Filters")
    
    # Geography filter - using list() to convert unique values to a list before sorting
    available_geographies = sorted(list(data['Geography'].unique()))
    selected_geography = st.sidebar.multiselect(
        "Select Geography",
        options=available_geographies,
        default=available_geographies
    )
    
    # Filter data based on selection
    filtered_data = data[data['Geography'].isin(selected_geography)]
    
    # Geographic Distribution
    st.subheader("Geographic Distribution")
    geography_counts = filtered_data['Geography'].value_counts()
    fig_geo = px.pie(
        values=geography_counts.values,
        names=geography_counts.index,
        title='Customer Distribution by Geography',
        hole=0.4
    )
    st.plotly_chart(fig_geo, use_container_width=True)
    
    # Churn Analysis
    st.subheader("Churn Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Churn
        fig_age = px.box(
            filtered_data,
            x='Exited',
            y='Age',
            title='Age Distribution by Churn Status'
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Balance vs Churn
        fig_balance = px.violin(
            filtered_data,
            x='Exited',
            y='Balance',
            title='Balance Distribution by Churn Status'
        )
        st.plotly_chart(fig_balance, use_container_width=True)
    
    # Customer Segments
    st.subheader("Customer Segments")
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score Distribution
        fig_credit = px.histogram(
            filtered_data,
            x='CreditScore',
            color='Exited',
            title='Credit Score Distribution',
            nbins=50
        )
        st.plotly_chart(fig_credit, use_container_width=True)
    
    with col2:
        # Active Members
        active_counts = filtered_data['IsActiveMember'].value_counts()
        fig_active = px.pie(
            values=active_counts.values,
            names=['Inactive', 'Active'],
            title='Active vs Inactive Members',
            hole=0.4
        )
        st.plotly_chart(fig_active, use_container_width=True)

    # Show filtered data summary
    st.subheader("Filtered Data Summary")
    summary_stats = filtered_data.describe()
    st.dataframe(summary_stats)

    # Download button for filtered data
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_churn_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    run_dashboard_page() 