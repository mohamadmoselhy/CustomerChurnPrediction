import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.data_utils import load_real_data

def run_analysis_page():
    st.title("Customer Churn Analysis")
    
    # Load the real data
    data = load_real_data()
    
    if data is None:
        st.error("Failed to load the dataset. Please check the file path.")
        return
    
    # Handle missing values in Geography
    data['Geography'] = data['Geography'].fillna('Unknown')
    
    st.info(f"Analyzing {len(data):,} customer records")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "Key Factors",
        "Customer Segments",
        "Financial Analysis"
    ])
    
    with tab1:
        st.subheader("Key Factors Influencing Customer Churn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution by Churn
            fig_age = px.box(
                data,
                x='Exited',  # Changed from 'Churn' to 'Exited'
                y='Age',
                title='Age Distribution by Churn Status',
                labels={'Exited': 'Customer Churned', 'Age': 'Customer Age'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Credit Score Distribution
            fig_credit = px.violin(
                data,
                x='Exited',  # Changed from 'Churn' to 'Exited'
                y='CreditScore',
                title='Credit Score Distribution by Churn Status',
                labels={'Exited': 'Customer Churned', 'CreditScore': 'Credit Score'}
            )
            st.plotly_chart(fig_credit, use_container_width=True)
        
        # Correlation Heatmap
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = data[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title='Feature Correlation Matrix',
            labels=dict(color='Correlation'),
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.subheader("Customer Segment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Geography Analysis
            geography_counts = data['Geography'].value_counts()
            fig_geo = px.pie(
                values=geography_counts.values,
                names=geography_counts.index,
                title='Customer Distribution by Geography',
                hole=0.4
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        
        with col2:
            # Gender Analysis
            gender_counts = data['Gender'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Customer Distribution by Gender',
                hole=0.4
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Products Distribution
        fig_products = px.histogram(
            data,
            x='NumOfProducts',
            color='Exited',  # Changed from 'Churn' to 'Exited'
            title='Number of Products per Customer',
            barmode='group'
        )
        st.plotly_chart(fig_products, use_container_width=True)
    
    with tab3:
        st.subheader("Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Balance Distribution
            fig_balance = px.histogram(
                data,
                x='Balance',
                color='Exited',  # Changed from 'Churn' to 'Exited'
                title='Balance Distribution by Churn Status',
                marginal='box'
            )
            st.plotly_chart(fig_balance, use_container_width=True)
        
        with col2:
            # Salary Distribution
            fig_salary = px.histogram(
                data,
                x='EstimatedSalary',
                color='Exited',  # Changed from 'Churn' to 'Exited'
                title='Salary Distribution by Churn Status',
                marginal='box'
            )
            st.plotly_chart(fig_salary, use_container_width=True)
        
        # Key Metrics
        st.subheader("Key Financial Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_balance = data['Balance'].mean()
            st.metric("Average Balance", f"${avg_balance:,.2f}")
        
        with col2:
            avg_salary = data['EstimatedSalary'].mean()
            st.metric("Average Salary", f"${avg_salary:,.2f}")
        
        with col3:
            churn_rate = (data['Exited'].mean() * 100)  # Changed from 'Churn' to 'Exited'
            st.metric("Churn Rate", f"{churn_rate:.1f}%")

if __name__ == "__main__":
    run_analysis_page() 