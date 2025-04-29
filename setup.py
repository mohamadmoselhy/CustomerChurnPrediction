from setuptools import setup, find_packages

setup(
    name="customer_churn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "streamlit>=1.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.5.0",
        "joblib>=1.1.0",
        "pathlib>=1.0.1",
        "imbalanced-learn==0.10.1",
        "pyyaml>=5.4.1",
        "pytest==7.3.1"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Customer Churn Prediction System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/customer-churn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 