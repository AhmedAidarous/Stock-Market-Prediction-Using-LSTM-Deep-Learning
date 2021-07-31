import os

import pandas_datareader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

import pandas
import pandas as pd
import plotly.express as px
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def cumulativeDifference(stock):
    """Method: Gets the cumulative return based on the stock prices"""
    for index in stock.columns[1:]:
        stock[index] = stock[index] / stock[index][0]

    return stock



def plotlyPlot(title, stock):
    """Method: Displays an interactive representation of given stock data in a
       line graph on your browser"""
    fig = px.line(title=title)
    for index in stock.columns[1:]:
        fig.add_scatter(x=stock['Date'], y=stock[index], name=index)

    fig.show()


def individualStock(priceDataFrame , volumeDataFrame, name):
    return pd.DataFrame({'Date':priceDataFrame['Date'], 'Close':priceDataFrame[name], 'Volume':volumeDataFrame[name]})


def tradingWindow(data, n):
    """Method: Creates a column that would form the price target prediction for a stock
                by getting the price for n days after each price"""
    dayShift = n
    data['target'] = data[['Adj Close']].shift(-dayShift)

    # Removes the last n rows to prevent errors
    data = data[:-n]
    return data


def LSTM (X_Train , X_Test):
    # Reshape the 1D to 3D arrays to feed in the mode, reshaping the training data.
    xTrain = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1] , 1))
    xTest = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1] , 1))

    # Building the LSTM deep neural network model
    inputLayer = keras.layers.Input(shape = (xTrain.shape[1] , xTrain.shape[2]))
    # return_sequences=True : Basically connects to the previous layer..
    hidden = keras.layers.LSTM(150, return_sequences=True) (inputLayer)
    hidden = keras.layers.LSTM(150, return_sequences=True)(hidden)
    hidden = keras.layers.LSTM(150, return_sequences=True)(hidden)
    # The output layer
    outputLayer = keras.layers.Dense(1 , activation='linear')(hidden)

    # Creating the model itself
    brainModel = keras.Model(inputs = inputLayer, outputs = outputLayer)
    brainModel.compile(optimizer = 'adam', loss = 'mse')
    brainModel.summary()

    # validation split would perform cross validation..
    brainModel.fit(X_Train , Y_Train, epochs = 20, batch_size = 32, validation_split = 0.2)

    return brainModel



def retrieveData(Start, End, Ticker):
    modifiedStart = pd.to_datetime(Start)
    modifiedEnd = pd.to_datetime(End)
    stock = web.DataReader(Ticker,'yahoo',modifiedStart, modifiedEnd)
    # Resets the date index to be able to use it as a column
    stock.reset_index(inplace=True, drop=False)
    return stock

# Retrieving the stockmarket data..
stockDataframe = retrieveData('2008-01-01','2020-10-01','GOOG')

stock = (stockDataframe[['Date' , 'Adj Close', 'Volume']])


priceVolumeTargetDataframe = tradingWindow(stock , 1)
print(priceVolumeTargetDataframe)


# normalizing the prices and volume with a feature range between 0-1
normalizeObj = MinMaxScaler(feature_range=(0,1))
priceVolumeTargetScaledDataframe = normalizeObj.fit_transform(priceVolumeTargetDataframe.drop(columns=['Date']))

# Feature : X , Target : Y
# This will get all the first two columns which are [Close , Volume]
X = priceVolumeTargetScaledDataframe[:,:2]
Y = priceVolumeTargetScaledDataframe[:,2:]


# Perform the trainTestSplit
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,train_size=0.65)



# Make Predictions
brainModel = LSTM(X_Train , X_Test)
predictions = brainModel.predict(X)


# Append the predicted values...
testPredictions = []
for elem in predictions:
    testPredictions.append(elem[0][0])

# Original closing prices
close = []
for i in priceVolumeTargetScaledDataframe:
    close.append(i[0])


dataFramePrediction = stock[1:][['Date']]
dataFramePrediction['Predictions'] = testPredictions
dataFramePrediction['Adj Close'] = close

plotlyPlot("LSTM Stock Performance Results" , dataFramePrediction)


