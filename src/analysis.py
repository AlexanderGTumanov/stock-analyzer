import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def retrieve_stock(ticker: str, start, end) -> pd.Series:
    df = yf.download(ticker, start = start, end = end, progress = False)
    return df['Close']

def log_returns(price_series: pd.Series) -> pd.Series:
    return np.log(price_series).diff().dropna()

def rolling_mean(series: pd.Series, window_size: int = None) -> pd.Series:
    if window_size is None:
        window_size = max(1, int(len(series) * 0.01))
    return series.rolling(window = window_size).mean()

def stationarity_test(series: pd.Series):
    """
    Perform stationarity tests on a time series using the Augmented Dickey-Fuller (ADF) 
    and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.

    Prints the test statistic, p-value, and critical values for both tests. Gives
    a basic prediction of stationarity based on 5% significance level.

    - ADF null hypothesis: the series has a unit root (non-stationary).
    - KPSS null hypothesis: the series is stationary around a constant.
    """
    adf_result = adfuller(series)
    print('ADF Test:')
    print(f'  Test Statistic: {adf_result[0]}')
    print(f'  p-value: {adf_result[1]}')
    print('  Critical Values:')
    for key, value in adf_result[4].items():
        print(f'    {key}: {value}')
    print('  Outcome: Stationary\n' if adf_result[1] < 0.05 else '  Outcome: Non-stationary\n')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = UserWarning)
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print('KPSS Test:')
    print(f'  Test Statistic: {kpss_result[0]}')
    print(f'  p-value: {kpss_result[1]}')
    print('  Critical Values:')
    for key, value in kpss_result[3].items():
        print(f'    {key}: {value}')
    print('  Outcome: Non-stationary' if kpss_result[1] < 0.05 else '  Outcome: Stationary')

def plot_series(series: pd.Series, title: str = None, xlabel: str = 'Date', ylabel: str = 'Value'):
    plt.figure(figsize = (10, 5))
    plt.plot(series.index, series.values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(series: pd.Series, lags: int = 40):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    for a given time series.
    """
    _, axes = plt.subplots(1, 2, figsize = (12, 4))
    plot_acf(series.dropna(), ax = axes[0], lags = lags, title = 'ACF')
    plot_pacf(series.dropna(), ax = axes[1], lags = lags, title = 'PACF', method = 'ywm')
    plt.tight_layout()
    plt.show()