import streamlit as st
import sys
from pathlib import Path
import logging
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the project root to Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from customer_churn.app.pages.prediction import run_prediction_page
from customer_churn.app.pages.analysis import run_analysis_page
from customer_churn.app.pages.dashboard import run_dashboard_page

# Create logs directory if it doesn't exist
log_dir = os.path.join(root_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio(
        'Select a page:',
        ['Prediction', 'Analysis', 'Dashboard']
    )
    
    if page == 'Prediction':
        run_prediction_page()
    elif page == 'Analysis':
        run_analysis_page()
    else:
        run_dashboard_page()

if __name__ == "__main__":
    main() 