import pickle
from pathlib import Path
from typing import Tuple, Any
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def save_model_artifacts(model: Any, scaler: Any, feature_names: list, model_dir: str) -> None:
    """Save model and related artifacts"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        'model_ensemble.pkl': model,
        'model_scaler.pkl': scaler,
        'model_features.pkl': feature_names
    }
    
    for filename, artifact in artifacts.items():
        with open(model_dir / filename, 'wb') as f:
            pickle.dump(artifact, f)
    
    logger.info(f"Saved model artifacts to {model_dir}")

def load_model_artifacts(model_dir: str) -> Tuple[Any, Any, list]:
    """Load model and related artifacts"""
    model_dir = Path(model_dir)
    
    try:
        with open(model_dir / 'model_ensemble.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(model_dir / 'model_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(model_dir / 'model_features.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        logger.info(f"Loaded model artifacts from {model_dir}")
        return model, scaler, feature_names
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate model performance metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_prob)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def train_and_save_model():
    """Train and save the model"""
    # Load your data
    # For demonstration, creating sample data
    np.random.seed(42)
    n_samples = 1000
    
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
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% churn rate
    })
    
    # Prepare features
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    
    # Convert categorical variables
    X = pd.get_dummies(X, columns=['Geography', 'Gender'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train model with categorical features enabled
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        enable_categorical=True  # Enable categorical features
    )
    model.fit(X_scaled, y)
    
    # Create models directory if it doesn't exist
    model_dir = Path('models/trained')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save artifacts
    with open(model_dir / 'model_ensemble.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(model_dir / 'model_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_dir / 'model_features.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    logger.info("Model trained and saved successfully") 