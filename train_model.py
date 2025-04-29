import os
import sys
from pathlib import Path

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from customer_churn.src.utils.model_utils import train_and_save_model

if __name__ == "__main__":
    print("Training model...")
    train_and_save_model()
    print("Model training completed!") 