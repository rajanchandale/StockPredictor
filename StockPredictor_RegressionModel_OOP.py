print("BUILDING ... ")

import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, scale


class DataFetcher:
    """
    Class responsible for fetching, computing, and preparing stock market data for a specified ticker
    """
    def __init__(self, ticker, start_date, end_date):
        """
        Constructor for DataFetcher.

        Parameters:
            - ticker (str): The stock symbol to fetch data for
            - start_date (str): The beginning date of the data range in "YYYY-MM-DD" format
            - end_date (str): The end date of the data range in "YYYY-MM-DD" format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def get_historical_data(self):
        """
        Fetches historical stock data for the given ticker from the Yahoo Finance API (yf)

        Returns:
            - pandas DataFrame: Historical stock data containing information such as Open, Close, High, Low, Volume etc
        """
        historical_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        return historical_data

    def compute_technical_indicators(self, data):
        """
        Calculates technical indicators from stock data

        Parameters:
            - data (pandas DataFrame): Historical stock data

        Returns:
            - pandas DataFrame: Data augmented with high-low percentage and percentage change columns
        """
        data['HIGH_LOW_PCT'] = (data['High'] - data['Low']) / data['Open'] * 100.0
        data['PCT_CHANGE'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

        return data

    def prepare_data(self, data):
        """
        Prepares the data for training and prediction by splitting it into training, testing, and prediction sets

        Parameters:
            - data (pandas DataFrame): Stock data with technical indicators

        Returns:
            - tuple: Training data (x_train, y_train), testing data (x_test, y_test), and prediction data (x_lately)
        """
        data_len = len(data)
        prediction_length = int(math.ceil(0.02*data_len))

        # Shifting the adjusted close values to create labels for prediction
        data['label'] = data['Adj Close'].shift(-prediction_length)

        cleaned_data = data.dropna()
        x_values = np.array(cleaned_data.drop(labels=['label'], axis=1))
        y_values = np.array(cleaned_data['label'])

        x_values = scale(x_values)

        x_lately = x_values[-prediction_length:]
        x_early = x_values[:-prediction_length]

        x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2)

        return x_train, x_test, y_train, y_test, x_lately


class ModelTrainer:
    """
    Class responsible for training, evaluating, and selecting regression models on stock data
    """
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor for ModelTrainer

        Parameters:
            - x_train, y_train (numpy arrays): Training data
            - x_test, y_test (numpy arrays): Testing data
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Models to be trained predefined here
        self.models = {
            "Linear Regression": LinearRegression(n_jobs=-1),
            "Quadratic Regression (2nd Order)": make_pipeline(PolynomialFeatures(2), Ridge()),
            "Quadratic Regression (3rd Order)": make_pipeline(PolynomialFeatures(3), Ridge())
        }

        self.confidences = {}

    def train(self):
        """
        Iteratively trains each model using training data and then evaluates its performance using the test data
        """
        for name, model in self.models.items():
            model.fit(self.x_train, self.y_train)
            self.confidences[name] = model.score(self.x_test, self.y_test)

    def evaluate_models(self):
        """
        Prints the performance (R^2 score) of each trained model
        """
        for name, confidence in self.confidences.items():
            print(f"{name} Confidence: {confidence:.4f}")

    def determine_best_model(self):
        """
        Determines the best performing model based on R^2 scores

        Returns:
            - sklearn model: The model with the highest R^2 score
        """
        most_accurate_model = max(self.confidences, key=self.confidences.get)

        return self.models[most_accurate_model]


class Visualiser:
    """
    Class responsible for visualising stock data and predictions
    """
    def __init__(self, stock_data, forecast_data, ticker):
        """
        Constructor for Visualiser

        Parameters:
            - stock_data (pandas DataFrame): Original stock data
            - forecast_data (numpy array): Predictions made by the best model
            - ticker (str): The stock symbol for which predictions were made
        """
        self.stock_data = stock_data
        self.forecast_data = forecast_data
        self.ticker = ticker

    def plot_predictions(self):
        """
        Plots historical stock prices along with predicted prices in a combined matplotlib plot
        """
        # Adjusting the prediction to align with the last actual data point
        adjustment = self.stock_data['Adj Close'].iloc[-1] - self.forecast_data[0]
        prediction_set = self.forecast_data + adjustment

        # Create new date indexes for the forecast data
        last_date = self.stock_data.iloc[-1].name
        new_dates = [last_date + datetime.timedelta(days=i) for i in range(1, len(prediction_set) + 1)]
        forecast_df = pd.DataFrame(data=prediction_set, index=new_dates, columns=['Forecast'])

        self.stock_data['Forecast'] = np.nan
        combined_data = self.stock_data.append(forecast_df)
        combined_data['Adj Close'].tail(500).plot(label=self.ticker + " HISTORIC SHARE PRICE", color="deepskyblue")
        combined_data['Forecast'].tail(500).plot(label=self.ticker + " PREDICTED SHARE PRICE", color="gold")

        plt.title(self.ticker + " MODEL")
        plt.legend(loc="upper right")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()


if __name__ == "__main__":
    # Entry point: User provides a stock ticker, and the program fetches data, trains models, and visualises predictions
    ticker = input("ENTER TICKER VALUE: \n").upper()

    start_date = (datetime.datetime.now() - datetime.timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    data_fetcher = DataFetcher(ticker, start_date, end_date)
    historical_data = data_fetcher.get_historical_data()
    technical_data = data_fetcher.compute_technical_indicators(historical_data)
    x_train, x_test, y_train, y_test, x_lately = data_fetcher.prepare_data(technical_data)

    model_trainer = ModelTrainer(x_train, y_train, x_test, y_test)
    model_trainer.train()
    model_trainer.evaluate_models()
    most_accurate_model = model_trainer.determine_best_model()

    prediction_set = most_accurate_model.predict(x_lately)

    visualiser = Visualiser(historical_data, prediction_set, ticker)
    visualiser.plot_predictions()
