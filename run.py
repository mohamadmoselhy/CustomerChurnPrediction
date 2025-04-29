import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from customer_churn.app.main import main

if __name__ == '__main__':
    main() 
