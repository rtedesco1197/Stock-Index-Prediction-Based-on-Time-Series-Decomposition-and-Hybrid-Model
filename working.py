#Implementing the CAL model (CEEMDAN-ARMA-LSTM) proposed by Pin Lv, Qinjuan Wu, Jia Xu, and Yating Shu (2021).
#Stock Index Prediction Based on Time Series Decomposition and Hybrid Model
#https://doi.org/10.3390/e24020146
import numpy as np
import pandas_datareader.data as web
import datetime as dt
from PyEMD import CEEMDAN
from statsmodels.tsa.stattools import adfuller
import pmdarima.arima as pmdarima
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.layers import GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot


if __name__ == "__main__":
    #Read in some stock data.
    start = dt.datetime(2007, 1, 1)
    end = dt.datetime(2020, 10, 1)
    symbol = 'VOO'  ###using Apple as an example
    source = 'yahoo'
    data = web.DataReader(symbol, source, start, end)
    prices = data['Adj Close'].values
    prices_step_ahead = np.delete(prices, 0, 0)
    close = np.delete(prices, -1, 0)
    #store test prices before CEEMDAN
    train_prices, test_prices = np.split(prices, [int(.90 * len(prices))])
    test_prices = np.delete(test_prices, -1)
    test_prices = test_prices[:-10,]
    test_prices = test_prices.reshape(1, len(test_prices))
    print(test_prices.shape)

    #CEEMDAN to extract IMF's (Intrinsic Mode Functions) and residual term from time series.
    ceemdan = CEEMDAN()
    cIMFs = ceemdan(prices)
    res = prices - np.sum(cIMFs, axis=0)
    IMFs_and_res = np.concatenate((cIMFs,res.reshape(1,len(res))))
    #Now we perform ADF test to see which IMF's are stationary. p-value <0.05= volatile/non-stationary
    NNfeatures = []
    ARMAfeatures= []
    for i in range(len(IMFs_and_res)):
        ADF_test = adfuller(IMFs_and_res[i])
        print('Column Name : ', i)
        print('p-value: %f' % ADF_test[1])
        if ADF_test[1]<0.05:
            tempNNarr = IMFs_and_res[i]
            NNfeatures.append(tempNNarr)
        else:
            tempARMAarr = IMFs_and_res[i]
            ARMAfeatures.append(tempARMAarr)

    #transpose before splitting train/test... probably a better way to this whole next step.
    NNfeatures = np.transpose(np.array(NNfeatures))
    ARMAfeatures = np.transpose(np.array(ARMAfeatures))
    # separate 90% of data for training, 10% for testing.
    NNtrain, NNtest = np.split(NNfeatures, [int(.90 * len(NNfeatures))])
    scaler = MinMaxScaler()
    NNtrain = np.transpose(NNtrain)
    NNtest = np.transpose(NNtest)
    ARMAtrain, ARMAtest = np.split(ARMAfeatures, [int(.90 * len(ARMAfeatures))])
    ARMAtrain = np.transpose(ARMAtrain)
    ARMAtest = np.transpose(ARMAtest)

    #Lag each of the 4 train/test variables for one step ahead prediction.
    NNtrain_step_ahead = np.delete(NNtrain, 1, 1)
    NNtrain = np.delete(NNtrain, -1, 1)
    NNtest_step_ahead = np.delete(NNtest, 1, 1)
    NNtest = np.delete(NNtest, -1, 1)
    ARMAtrain_step_ahead = np.delete(ARMAtrain, 1, 1)
    ARMAtrain = np.delete(ARMAtrain, -1, 1)
    ARMAtest_step_ahead = np.delete(ARMAtest, 1, 1)
    ARMAtest = np.delete(ARMAtest, -1, 1)
    ARMA_IDF_preds = []

    #Build ARMA model to predict stationary (non-volatile) IMFs
    for i in range(ARMAtrain.shape[0]):
        #Since Python only has auto arima, I set d=0 to optimize an equivalent arma(p,q) model.
        model = pmdarima.auto_arima(ARMAtrain[i], max_d=0, max_D=0, d=0, D=0, seasonal=False)
        tempPreds = model.predict(n_periods=ARMAtest.shape[1] - 10)
        print(tempPreds)
        ARMA_IDF_preds.append(tempPreds)


    #Build LSTM neural network to predict non-stationary (volatile) IMFs
    m = Sequential()
    m.add(LSTM(units=128, input_shape=(10, 1), return_sequences=True))
    #m.add(Dropout(0.1))
    m.add(LSTM(units=64, activation='relu', return_sequences=True))
    #m.add(Dropout(0.1))
    m.add(LSTM(units=16, activation='tanh', return_sequences=True))
    m.add(Dense(units=1))
    m.compile(optimizer='adam', loss='mean_squared_error')
    # reshape input to be [samples, time steps, features]
    NNtrain = np.transpose(NNtrain)
    NNtrain_step_ahead = np.transpose(NNtrain_step_ahead)
    NNtest = np.transpose(NNtest)
    NNtest_step_ahead = np.transpose(NNtest_step_ahead)

    LSTM_IDF_preds = []
    for i in range(NNtest.shape[1]):
        generator = TimeseriesGenerator(NNtrain[:, i], NNtrain_step_ahead[:, i], length=10, batch_size=128)
        history = m.fit(generator, epochs=50, verbose=1)
        generator_test = TimeseriesGenerator(NNtest[:, i], NNtest_step_ahead[:, i], length=10, batch_size=128)
        yhat_test = m.predict(generator_test, verbose=0)
        yhat_test = GlobalAveragePooling1D()(yhat_test)
        LSTM_IDF_preds.append(yhat_test)

    #print(np.shape(LSTM_IDF_preds))
    #CAL Predictions: We sum forecasts across each previous model for IMFs and residue
    #Now we concatenate our predictions and sum across each IDF from the ARMA and LSTM models.
    sum_LSTM_preds = np.sum(LSTM_IDF_preds, axis=0)
    print(np.shape(LSTM_IDF_preds))
    print(np.shape(ARMA_IDF_preds))

    sum_ARMA_preds= np.asarray(ARMA_IDF_preds)
    sum_LSTM_preds = np.transpose(sum_LSTM_preds)

    if sum_ARMA_preds.shape[0] != 1:
        sum_ARMA_preds = np.sum(ARMA_IDF_preds, axis=0)
    print(sum_ARMA_preds.shape)
    print(sum_LSTM_preds.shape)

    #Finally, sum our LSTM and ARMA preds.
    finalpreds = np.add(sum_LSTM_preds, sum_ARMA_preds)
    print(finalpreds.shape)
    print(finalpreds)
    print(test_prices)
    #How'd we do? RMSE is ~30, given that VOO is $400 a share, our model is roughly ~10% off on average
    RMSE = np.sqrt(mean_squared_error(finalpreds, test_prices))
    print(RMSE)

    #Plot forecast vs actual
    pyplot.plot(np.reshape(test_prices, (test_prices.shape[1], 1)), label='Actual')
    pyplot.plot(np.reshape(finalpreds, (finalpreds.shape[1], 1)), label='Predicted')
    pyplot.legend()
    pyplot.show()
