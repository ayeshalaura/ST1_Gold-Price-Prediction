# Gold Price Prediction

## Overview

This project uses time-series regression techniques to predict the **Adjusted Close** price of a Gold ETF. Leveraging economic indicators from the **Gold Price Prediction Dataset** on [Kaggle](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset), the project aims to model and predict gold prices using various statistical and machine learning approaches.

### Goals:
- Predict the **Adjusted Close** value of a Gold ETF.
- Implement feature engineering, model evaluation, and selection.
- Select and deploy the best-performing model for real-time predictions.

## Requirements

Install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

## Usage

### 1. Run the Web Application for Predictions

You can immediately try a demo of the Flask-based web application to make predictions using the latest available trained model:

```
python src/webapp/app.py
```

Navigate to `http://127.0.0.1:5000/` in your browser to view the app. The app will display the predicted **Adjusted Close** value based on the most recent economic data available.

### 2. Run the Data Pipeline

If you'd like to process, analyze, and train models on the data, run the main pipeline with the following command:

```
python src/main.py
```

**Note**: This script performs data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and serialization. The pipeline may take **5-15 minutes** to complete, depending on your system's specifications.

## Problem Statement

The objective is to develop a reliable model to predict the Adjusted Close price of a Gold ETF by analyzing various economic indicators. By leveraging machine learning and time-series analysis, this project seeks to establish a robust model that captures economic influences on gold prices, which can be used to forecast future price movements.
