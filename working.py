#Implementing the CAL model (CEEMDAN-ARMA-LSTM) proposed by Pin Lv, Qinjuan Wu, Jia Xu, and Yating Shu (2021).
#Stock Index Prediction Based on Time Series Decomposition and Hybrid Model
#https://doi.org/10.3390/e24020146

import numpy as np
import yfinance as yf
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
import random

random.seed(10)

class StockData:
    def __init__(self, start_datetime, end_datetime, ticker):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.ticker = ticker
        self.load_data()

    def load_data(self):
        data = yf.download(self.ticker, self.start_datetime, self.end_datetime)
        self.prices = data['Adj Close'].values
        train_prices, test_prices = np.split(self.prices, [int(.90 * len(self.prices))])
        test_prices = np.delete(test_prices, -1)
        test_prices = test_prices[:-10]
        self.test_prices = test_prices.reshape(1, len(test_prices))

start = dt.datetime(2007, 1, 1)
end = dt.datetime(2023, 10, 1)
symbol = 'AMZN'
stock_data = StockData(start, end, symbol)
print(stock_data.test_prices)

class CAL_model:
  def __init__(self, prices, test_prices):
    self.prices = prices
    self.test_prices = test_prices
    self.load_data()

  def preprocessing(self):
     if __name__ == "__main__":
        #CEEMDAN to extract IMF's (Intrinsic Mode Functions) and residual term from time series.
        ceemdan = CEEMDAN()
        cIMFs = ceemdan(prices)
        print(cIMFs.shape)
        print(np.sum(cIMFs, axis=0))
        res = prices - np.sum(cIMFs, axis=0)
        IMFs_and_res = np.concatenate((cIMFs,res.reshape(1,len(res))))
        print(IMFs_and_res)
        print(IMFs_and_res.shape)
        #Now perform ADF test to see which IMF's are stationary. p-value <0.05= volatile/non-stationary
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
        self.NNtrain_step_ahead = np.delete(NNtrain, 1, 1)
        self.NNtrain = np.delete(NNtrain, -1, 1)
        self.NNtest_step_ahead = np.delete(NNtest, 1, 1)
        self.NNtest = np.delete(NNtest, -1, 1)

        print("AYOOOO IDIOT")
        print(NNtest)

        #ARMAtrain_step_ahead = np.delete(ARMAtrain, 1, 1)
        self.ARMAtrain = np.delete(ARMAtrain, -1, 1)
        self.ARMAtest_step_ahead = np.delete(ARMAtest, 1, 1)
        self.ARMAtest = np.delete(ARMAtest, -1, 1)


  def fit_model(self):
      #Build ARMA model to predict stationary (non-volatile) IMFs
      self.ARMA_IDF_preds = []
      for i in range(fit_test.ARMAtrain.shape[0]):
        #Since Python only has auto arima, I set d=0 to optimize an equivalent arma(p,q) model.
        model = pmdarima.auto_arima(fit_test.ARMAtrain[i], max_d=0, max_D=0, d=0, D=0, seasonal=False)
        tempPreds = model.predict(n_periods=fit_test.ARMAtest.shape[1] - 10)
        print(tempPreds)
        fit_test.ARMA_IDF_preds.append(tempPreds)

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

      self.LSTM_IDF_preds = []
      for i in range(NNtest.shape[1]):
        generator = TimeseriesGenerator(NNtrain[:, i], NNtrain_step_ahead[:, i], length=10, batch_size=128)
        history = m.fit(generator, epochs=50, verbose=1)
        generator_test = TimeseriesGenerator(NNtest[:, i], NNtest_step_ahead[:, i], length=10, batch_size=128)
        yhat_test = m.predict(generator_test, verbose=0)
        yhat_test = GlobalAveragePooling1D()(yhat_test)
        fit_test.LSTM_IDF_preds.append(yhat_test)

  def predictions(self):
      sum_LSTM_preds = np.sum(fit_test.LSTM_IDF_preds, axis=0)
      print(np.shape(fit_test.LSTM_IDF_preds))
      print(np.shape(fit_test.ARMA_IDF_preds))

      sum_ARMA_preds= np.asarray(fit_test.ARMA_IDF_preds)
      sum_LSTM_preds = np.transpose(sum_LSTM_preds)
      #Finally, sum our LSTM and ARMA preds.
      self.finalpreds = np.add(sum_LSTM_preds, sum_ARMA_preds)
      print(fit_test.finalpreds.shape)
      print(fit_test.finalpreds)
      print(fit_test.test_prices)
      #How'd we do? RMSE is ~30, given that VOO is $400 a share, our model is roughly ~10% off on average
      RMSE = np.sqrt(mean_squared_error(fit_test.finalpreds, fit_test.test_prices))
      print(RMSE)
      #Plot forecast vs actual
      pyplot.plot(np.reshape(fit_test.test_prices, (fit_test.test_prices.shape[1], 1)), label='Actual')
      pyplot.plot(np.reshape(fit_test.finalpreds, (fit_test.finalpreds.shape[1], 1)), label='Predicted')
      pyplot.legend()
      pyplot.show()

cal = CAL_model(stock_data.prices, stock_data.test_prices)
cal
fit_test=np.transpose(cal)
stock_data.test_prices = np.transpose(stock_data.test_prices)

#Detect large predicted price movements?
def trade_strat(initial_shares):
    sd_pred = np.std(fit_test)
    buy_sell_signal = np.zeros(len(fit_test))
    start_position = stock_data.test_prices[0]*initial_shares
    total_shares=initial_shares
    timeframe = len(stock_data.test_prices)
    print(timeframe)
    timeframe_months = round(len(stock_data.test_prices)/30,1)


    for i in range(1,len(fit_test)):
        if fit_test[i] > fit_test[i-1] - sd_pred*3 and total_shares < 150:
            buy_sell_signal[i] = 1
            total_shares -= 10
            print("Sell!")
        if fit_test[i] < fit_test[i-1] - sd_pred*3 and total_shares>=10:
            buy_sell_signal[i] = -1
            total_shares += 10
            print("Buy!")

    end_position = total_shares*stock_data.test_prices[len(stock_data.test_prices) - 1]
    if_held_initial = initial_shares * stock_data.test_prices[len(stock_data.test_prices) - 1]
    gain_over_hold_strat = end_position - if_held_initial
    profit_loss = end_position - start_position
    profit_loss_pct = np.round((end_position-start_position)/start_position*100,2)

    result = f"Timeframe: {timeframe} days / {timeframe_months} months, End Position is ${end_position} with {total_shares} total shares. Profit_loss is ${profit_loss} which is a {profit_loss_pct}% gain/loss, This got you ${gain_over_hold_strat} more than holding initial {initial_shares} shares."
    return result

final = trade_strat(100)
final
