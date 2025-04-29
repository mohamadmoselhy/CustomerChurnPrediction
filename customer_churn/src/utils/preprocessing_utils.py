import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features"""
    df = df.copy()
    
    interactions = {
        'Balance_Per_Product': lambda x: x['Balance'] / (x['NumOfProducts'] + 1),
        'Credit_Balance_Ratio': lambda x: x['CreditScore'] / (x['Balance'] + 1),
        'Salary_Balance_Ratio': lambda x: x['EstimatedSalary'] / (x['Balance'] + 1),
        'Age_Product_Interaction': lambda x: x['Age'] * x['NumOfProducts'],
        'Tenure_Balance_Interaction': lambda x: x['Tenure'] * np.log1p(x['Balance']),
        'Credit_Age_Interaction': lambda x: x['CreditScore'] * x['Age']
    }
    
    for name, func in interactions.items():
        df[name] = func(df)
    
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features"""
    df = df.copy()
    
    # Customer value score
    df['Customer_Value_Score'] = (
        df['Balance'] * df['IsActiveMember'] * df['Tenure']
    ) / (df['NumOfProducts'] + 1)
    
    # Risk score
    df['Risk_Score'] = (
        df['CreditScore'] * df['IsActiveMember'] * df['HasCrCard']
    ) / (df['NumOfProducts'] + 1)
    
    return df

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoders: Dict[str, LabelEncoder] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features"""
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_columns:
        if col not in encoders:
            encoders[col] = LabelEncoder()
        
        df[col] = encoders[col].fit_transform(df[col])
    
    return df, encoders 