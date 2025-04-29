from setuptools import setup, find_packages

setup(
    name="customer_churn",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'streamlit',
        'plotly',
        'joblib',
        'imbalanced-learn',
        'pyyaml',
    ],
) 