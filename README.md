# StockPredictor

I have always taken a keen interest in the stock market and wanted to create an algorithm to predict the performance of a stock based on either a linear or quadratic regression model. This project served as an introduction to Machine Learning and FinTech.

StockPredictor is a Python program that fetches historical stock data, trains multiple regression models to predict future stock prices, and visualises the predictions in a graph.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Code Overview](#code-overview)
- [Approach and Logic](#approach-and-logic)
- [Usage](#usage)

## Features

- Fetches historical stock data for the past five years using the Yahoo Finance API
- Computes technical indicators from the fetched stock data
- Trains and evaluates multiple regression models on the historical data
- Visualises historical stock prices along with the model's predictions

## Prerequisites

To install the required libraries, run the following command:
```
pip install -r requirements.txt
```

## Code Overview

- **Data Fetcher:** Responsible for fetching, computing, and preparing stock market data for a specified ticker
- **Model Trainer:** Trains, evaluates, and selects regression models on the fetched stock data
- **Visualiser:** Visualises the historical and predicted stock prices in a graph

## Approach and Logic

### 1. Data Collection and Preprocessing:
The program starts by fetching historical stock data using the yfinance library, which provides a reliable interface to Yahoo Finance's data. We collect the stock data for the past five years. 

Once the data is fetched, it is essential to create features that can help in predictive modelling. Two technical indicators are calculated:
- **High_Low_Pct:** Calculates the percentage between the High and Low prices of the stock for a given day
- **Pct_Change:** Represents the percentage change between the Opening and Closing prices

This indicators provide an insight to the stock's daily volatility and its intra-day sentiment.

### 2. Model Training and Evaluation:

Predicting stock prices is a regression problem. Different regression models are used to train the preprocessed data. The regression models evaluated are:
- Linear Regression
- Quadratic Regression (2nd Order)
- Quadratic Regression (3rd Order)

Each model is trained on the historical stock data, and its performance is evaluated using the R^2 score on a test set. The model with the best performance is then chosen for making predictions.

### 3. Visualisation:

Once predictions are made, it is essential to visually represent these in relation to historical prices. This helps in understanding how the model perceives future price movements.

The prediction is adjusted to align with the last actual data point for continuity in the plot. The historical stock prices and the predicted prices are then plotted using matplotlib, providing a comprehensive view of the stock's past performance and its potential future trajectory. 

## Usage

1. Navigate to the directory containing StockPredictor_RegressionModel_OOP.py
2. Run the program using the command:
```
python StockPredictor_RegressionModel_OOP.py
```
3. Enter the desired stock ticker symbol when prompted
4. The program will fetch the data, train the models, evaluate their performance, and display a plot visualising the predictions against the actual historical prices.
