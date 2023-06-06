


import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
import numpy as np

from matplotlib import pyplot as plt

def stationarity_test(data):
    stationary = False

    while not stationary:
        p_value = adfuller(data)[1]
        if p_value <= 0.05:
            stationary = True
        else:
            data = [data[i] - data[i - 1] for i in range(1, len(data))]
    return data


def find_best_order_ARIMA( data, max_pdq):
    d_values = p_values = q_values = range(0,max_pdq)
    min_aic = float('inf')
    best_order = (0, 0, 0)

    for p in p_values:
        for d in d_values:
            for q in q_values:

                model = ARIMA(data,order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic

                if aic < min_aic:
                    min_aic = aic
                    best_order = (p, d, q)
    return best_order


# https://github.com/ManojKumarMaruthi/Time-Series-Forecasting/blob/master/Modeling(Auto%20Regression%2CARMA%2CARIMA%2CSARIMA).ipynb
# https://moodledidattica.univr.it/pluginfile.php/1273389/mod_resource/content/0/TimeSeries.ipynb

if __name__ == "__main__":

    stocks = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'AAPL', 'JNJ', 'BRK-A', 'META', 'PG', 'V',
              'JPM', 'XOM', 'KO', 'WMT', 'PFE', 'HD', 'MA', 'NKE', 'DIS', 'MCD', 'UNH']

    for stock in stocks:
        data = pd.read_csv(f'./data_feed/{stock}.csv')

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        data = stationarity_test(data['Adj Close'])


        train, test = train_test_split(data, test_size=0.2, shuffle=False)


        # Autoregressive
        # model = AutoReg(train,lags=5)
        # model_fit = model.fit()
        # predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
        # # plot results
        # plt.figure(figsize=(12, 6))
        # plt.plot(test, label='real')
        # plt.plot(predictions, color='red', label='forecast')
        # plt.title(stock)
        # plt.legend()
        # plt.show()

        # ARIMA
        best_order = find_best_order_ARIMA(train, 3)

        model = ARIMA(train, order=best_order)
        model_fit = model.fit()

        print(f'STOCK: {stock} \t BEST ORDER: {best_order}')

        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
        # plot results
        plt.figure(figsize=(12, 6))
        plt.plot(test, label='real')
        plt.plot(predictions, color='red', label='forecast')
        plt.title(stock)
        plt.legend()
        plt.show()





