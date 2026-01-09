# Boston House Price Prediction

A machine learning project that predicts house prices using the Boston Housing dataset. This project implements a linear regression model to estimate median property values based on various housing features.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Key Results](#key-results)

## Overview

This project applies machine learning techniques to predict house prices in the Boston area. The model is built using Python and scikit-learn, implementing a Linear Regression algorithm with comprehensive data analysis and evaluation metrics.

### Objectives

- Load and explore the Boston Housing dataset
- Perform exploratory data analysis (EDA)
- Preprocess and scale the data
- Train a linear regression model
- Evaluate model performance
- Make predictions on new data
- Serialize the model for deployment

## Dataset

The Boston Housing dataset contains information collected by the U.S. Census Service regarding housing in the Boston, Massachusetts area.

### Features (13 attributes)

- **CRIM** - Per capita crime rate by town
- **ZN** - Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS** - Proportion of non-retail business acres per town
- **CHAS** - Charles River dummy variable (1 if bounds river; 0 otherwise)
- **NOX** - Nitric oxides concentration (parts per 10 million)
- **RM** - Average number of rooms per dwelling
- **AGE** - Proportion of owner-occupied units built prior to 1940
- **DIS** - Weighted distances to five Boston employment centres
- **RAD** - Index of accessibility to radial highways
- **TAX** - Full-value property-tax rate per $10,000
- **PTRATIO** - Pupil-teacher ratio by town
- **B** - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- **LSTAT** - Percentage of lower status of the population

### Target Variable

- **MEDV** (renamed as **PRICE**) - Median value of owner-occupied homes in $1000s

## Features

- ✓ Comprehensive exploratory data analysis with visualizations
- ✓ Correlation analysis and regression plots
- ✓ Data preprocessing and standardization using StandardScaler
- ✓ Train-test split (70-30 split)
- ✓ Linear Regression model training
- ✓ Detailed performance metrics (MSE, MAE, R² score)
- ✓ Residual analysis for model validation
- ✓ Model serialization with pickle for deployment
- ✓ Prediction on new data

## Project Structure

```
ML-Predictions/
├── BostonHouse.ipynb           # Main Jupyter notebook with complete analysis
├── BostonHousing.csv           # Boston Housing dataset
├── regmodel.pkl                # Serialized trained model (generated after running notebook)
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries listed below

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `BostonHouse.ipynb`

3. Run cells sequentially (Shift + Enter) to:
   - Load and explore the data
   - Perform EDA with visualizations
   - Train the linear regression model
   - Generate performance metrics
   - Make predictions on new data

### Making Predictions

To make predictions using the trained model:

```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the scaler and model
pickled_model = pickle.load(open('regmodel.pkl', 'rb'))

# Prepare your data (13 features in the same order as training)
new_data = np.array([...]).reshape(1, -1)  # Shape: (1, 13)

# Scale the data using the fitted scaler
scaled_new_data = scaler.transform(new_data)

# Make prediction
predicted_price = pickled_model.predict(scaled_new_data)
print(f"Predicted Price: ${predicted_price[0] * 1000:,.2f}")  # Price is in thousands
```

## Model Performance

### Evaluation Metrics

The model is evaluated using the following metrics:

- **Mean Squared Error (MSE)** - Measures the average squared difference between predicted and actual values
- **Mean Absolute Error (MAE)** - Measures the average absolute difference between predicted and actual values
- **R² Score (Coefficient of Determination)** - Indicates how well the model explains the variance in the target variable

### Performance Analysis

The project includes:

1. **Residual Analysis** - Kernel Density Estimation (KDE) plot of residuals to check for normal distribution
2. **Residuals vs Predictions** - Scatter plot to verify uniform distribution of errors
3. **Feature Relationships** - Regression plots showing relationships between key features and price:
   - RM (rooms) vs Price - Strong positive correlation
   - LSTAT (lower status %) vs Price - Strong negative correlation
   - CHAS (Charles River) vs Price - Categorical feature impact
   - PTRATIO (pupil-teacher ratio) vs Price - Negative correlation
   - CRIM (crime rate) vs Price - Negative correlation

## Key Results

- **Model Type**: Linear Regression
- **Training Data**: 70% of dataset (354 samples)
- **Test Data**: 30% of dataset (152 samples)
- **Data Preprocessing**: StandardScaler normalization applied
- **Model Output**: Predictions saved and model serialized using pickle for production deployment

### Example Prediction

The notebook demonstrates making a prediction on the first data point in the dataset and compares it with the actual price.

## Deployment

The trained model is serialized and saved as `regmodel.pkl`. This allows the model to be:

- Loaded in other applications
- Deployed to production environments
- Used without retraining
- Shared with other team members

## Technologies Used

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning library
- **Jupyter Notebook** - Interactive computing environment

## Author

Created as a machine learning demonstration project for house price prediction using linear regression.

## License

This project is provided for educational purposes.

## References

- Boston Housing Dataset - Originally from UCI Machine Learning Repository
- scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/