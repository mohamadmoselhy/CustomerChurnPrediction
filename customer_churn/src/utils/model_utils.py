import pickle
from pathlib import Path
from typing import Tuple, Any
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

def save_metrics_to_file(metrics: dict, output_file: str) -> None:
    """Save model metrics to a text file"""
    with open(output_file, 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("=======================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.3f}\n")
    
    logger.info(f"Saved metrics to {output_file}")

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate model performance metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_prob)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    # Save metrics to file
    save_metrics_to_file(metrics, 'models/trained/model_metrics.txt')
    
    return metrics

def load_data():
    """Load the dataset from CSV file"""
    file_path = r"CustomerChurnPrediction\Data\DataSet Before Cleanig.csv"
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the data for modeling"""
    # Drop any rows with missing values
    data = data.dropna()
    
    # Separate features and target
    X = data.drop(['Exited', 'CustomerId', 'Surname', 'RowNumber'], axis=1, errors='ignore')
    y = data['Exited']
    
    return X, y

def train_and_save_model():
    """Train and save the model with improvements for class imbalance"""
    # Load the real data
    data = load_data()
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Convert categorical variables
    categorical_features = ['Geography', 'Gender']
    X_encoded = pd.get_dummies(X, columns=categorical_features)
    
    # Get the list of feature names after encoding
    feature_names = X_encoded.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with SMOTE and XGBoost
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__scale_pos_weight': [1, 3, 5],  # To handle class imbalance
        'classifier__min_child_weight': [1, 3],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',  # Optimize for F1-score
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    logger.info("Starting model training with grid search...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_pipeline = grid_search.best_estimator_
    
    # Get the scaler from the pipeline
    scaler = best_pipeline.named_steps['scaler']
    
    # Get the final model from the pipeline
    model = best_pipeline.named_steps['classifier']
    
    # Create models directory if it doesn't exist
    model_dir = Path('models/trained')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save artifacts
    save_model_artifacts(model, scaler, feature_names, str(model_dir))
    
    # Calculate and log performance metrics
    y_pred = model.predict(scaler.transform(X_test))
    y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    
    logger.info("Model performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.3f}")
    
    logger.info("Model trained and saved successfully")
    return model, scaler, feature_names 