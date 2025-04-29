import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def preprocess_data(input_file: str, output_file: str = "DataAfterDataPreProcessing.csv"):
    """
    Preprocess the customer churn dataset with the following transformations:
    - Convert categorical variables to numerical
    - Create range categories for numerical variables
    - Add one-hot encoding for geography
    - Add gender label encoding
    """
    try:
        # Read the input data
        df = pd.read_csv(input_file)
        logger.info(f"Successfully loaded data with shape: {df.shape}")

        # Ensure all required columns exist
        required_columns = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary', 'Exited'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create gender label encoding
        df['genderlabel'] = df['Gender'].map({'Female': 0, 'Male': 1})

        # Create one-hot encoding for geography
        df['geographyfrance'] = (df['Geography'] == 'France').astype(int)
        df['geographygermany'] = (df['Geography'] == 'Germany').astype(int)
        df['geographyspain'] = (df['Geography'] == 'Spain').astype(int)

        # Create range categories for numerical variables
        # CreditScore ranges
        df['creditscorerange'] = pd.cut(df['CreditScore'], 
                                      bins=[0, 400, 500, 600, 700, 850],
                                      labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])

        # Balance ranges
        df['balancerange'] = pd.qcut(df['Balance'], 
                                   q=5,
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # EstimatedSalary ranges
        df['estimatedsalaryrange'] = pd.qcut(df['EstimatedSalary'], 
                                           q=5,
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Tenure ranges
        df['tenurerange'] = pd.cut(df['Tenure'],
                                 bins=[0, 2, 4, 6, 8, 10],
                                 labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])

        # Age skewed transformation
        df['ageskewed'] = np.log1p(df['Age'])

        # Save the preprocessed data
        output_path = Path(__file__).parent.parent.parent.parent / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to: {output_path}")

        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "Churn_Modelling.csv"  # Replace with your actual input file name
    preprocess_data(input_file) 