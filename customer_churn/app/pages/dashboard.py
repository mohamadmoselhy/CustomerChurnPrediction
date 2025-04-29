import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
from src.utils.data_utils import load_real_data
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

def display_model_metrics(y_true, y_pred, y_prob):
    """Display model metrics in a visually appealing way"""
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [precision, recall, f1, roc_auc]
    })
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", f"{precision:.3f}")
    with col2:
        st.metric("Recall", f"{recall:.3f}")
    with col3:
        st.metric("F1 Score", f"{f1:.3f}")
    with col4:
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    
    # Create confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    roc_fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    st.plotly_chart(roc_fig)

def run_dashboard_page():
    st.title("Customer Churn Dashboard")
    
    # Load the real data
    data = load_real_data()
    
    if data is None:
        st.warning("No data available. Please ensure the data file exists.")
        return
    
    # Load model artifacts
    try:
        model_dir = Path("models")
        model, scaler, feature_names = load_model_artifacts(model_dir)
        
        # Prepare data for prediction
        X = data[feature_names]
        X_scaled = scaler.transform(X)
        y_true = data['Exited']
        
        # Get predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        # Display metrics
        st.header("Model Performance Metrics")
        display_model_metrics(y_true, y_pred, y_prob)
        
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