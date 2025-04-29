import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from file"""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def check_data_quality(df: pd.DataFrame) -> Dict:
    """Perform data quality checks"""
    quality_report = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'unique_values': df.nunique().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return quality_report

def prepare_input_data(input_dict: Dict) -> pd.DataFrame:
    """Convert input dictionary to DataFrame with proper format"""
    df = pd.DataFrame([input_dict])
    required_columns = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary'
    ]
    
    # Validate input columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def validate_numeric_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate numeric input ranges"""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")

def load_real_data():
    """Load the actual customer churn dataset"""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        data_path = project_root / "DataAfterDataPreProcessing.csv"
        
        if not data_path.exists():
            st.error(f"Data file not found at: {data_path}")
            return None
            
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None 