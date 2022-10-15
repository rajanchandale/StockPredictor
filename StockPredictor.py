print("BUILDING ... ")

#required libraries
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as pd_web
from pandas import *
from sklearn import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.neighbors import *

#start and end dates of historic close values of the stock 
startDate = datetime.datetime(2010,1,1)
endDate = datetime.datetime.now()

#stock ticker
ticker = input("PLEASE ENTER TICKER:\n").upper()

#acquires historic stock close prices from public yahoo api between the two dates
data = pd_web.DataReader(ticker,'yahoo',startDate,endDate)

closePx = data['Adj Close']
#moving average
mavg = closePx.rolling(window=100).mean()

dataReg = data.loc[:,['Adj Close','Volume']]
dataReg['HL_PCT'] = (data['High'] - data['Low']) / data['Open'] * 100.0
dataReg['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

#removes missing values to clean data
dataReg.fillna(value=-99999, inplace = True)

#finding the 1% of data for our prediction
predictionOut = int(math.ceil(0.01 * len(dataReg)))

#Specifying the data we need for our model (closing prices)
predictionColumn = 'Adj Close'

dataReg['label'] = dataReg[predictionColumn].shift(-predictionOut)
x = np.array(dataReg.drop(['label'], 1))

#x values need to be scaled to allow for same distribution in linear regression
x = preprocessing.scale(x)

#finding late x and early x for model generation and evaluation
x_lately = x[-predictionOut:]
x = x[:-predictionOut]

y = np.array(dataReg['label'])
y = y[:-predictionOut]

#splitting train and test with cross validation train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)

#creating linear regeression Model
linearRegressionModel = LinearRegression(n_jobs=-1)
linearRegressionModel.fit(xTrain,yTrain)

#creating quadratic regression model 1 
quadModel1 = make_pipeline(PolynomialFeatures(2),Ridge())
quadModel1.fit(xTrain,yTrain)

#creating quadratic regression model 2
quadModel2 = make_pipeline(PolynomialFeatures(3),Ridge())
quadModel2.fit(xTrain,yTrain)

#Testing the accuracy of each model 
confidenceLin = linearRegressionModel.score(xTest,yTest)
confidenceQuad1 = quadModel1.score(xTest,yTest)
confidenceQuad2 = quadModel2.score(xTest,yTest)

#finding the most accurate model for this prediction
accurateModel = linearRegressionModel
mostConfident = confidenceLin
if confidenceQuad1 > mostConfident:
    accurateModel = quadModel1
elif confidenceQuad2 > mostConfident:
    accurateModel = quadModel2

#building prediction model 
prediction_set = accurateModel.predict(x_lately)
dataReg['Forecast'] = np.nan

lastDate = dataReg.iloc[-1].name
lastUnix = lastDate
nextUnix = lastUnix + datetime.timedelta(days=1)

for i in prediction_set:
    nextDate = nextUnix
    nextUnix += datetime.timedelta(days=1)
    dataReg.loc[nextDate] = [np.nan for _ in range(len(dataReg.columns)-1)]+[i]

dataReg['Adj Close'].tail(500).plot(label = ticker + " HISTORIC SHARE PRICE",color = "deepskyblue")
dataReg['Forecast'].tail(500).plot(label = ticker + " PREDICTED SHARE PRICE",color = "gold")
plt.title(ticker + " MODEL")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
                                                    

