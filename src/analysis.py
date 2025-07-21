import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def retrieve_stock(ticker: str, start, end) -> pd.Series:
    df = yf.download(ticker, start = start, end = end, progress = False)
    return df['Close'].squeeze()

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

    Parameters
    ----------
    series : pd.Series
        The complete time series to test for stationarity.
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

# def plot_series(series: pd.Series, model = None, k = 2, horizon = 0):
#     """
#     Plot a time series along with a model fit and forecast (if provided).

#     Parameters
#     ----------
#     series : pd.Series
#         The complete time series data to plot. If a model fit is provided, the series may extend
#         beyond its range.
#     model : optional
#         A fitted time series model (e.g., from `arch` or similar library). If provided, the function
#         will plot the fitted mean and ±k standard deviation band from the model, and optionally 
#         a forecast.
#     k : int, default = 2
#         Number of standard deviations for the confidence band.
#     horizon : int, default = 0
#         Number of periods to forecast beyond the end of the fit. If zero, no forecast is shown.
#     """
#     index = series.index
#     plt.figure(figsize = (10, 5))
#     if model:
#         fit_len = len(model.model.y)
#         fit_index = index[:fit_len]
#         extra_index = index[fit_len - 1:]
#         plt.plot(fit_index, series.loc[fit_index], label = 'Original Series (fit)', color = 'blue')
#         if len(extra_index) > 0:
#             plt.plot(extra_index, series.loc[extra_index], label = 'Original Series (not fit)', color = "#5972CD")
#         std = model.conditional_volatility
#         mean = pd.Series(model.model.y, index = fit_index) - model.resid
#         if horizon:
#             fit_end = fit_index[-1]
#             forecast = model.forecast(horizon = horizon, start = fit_end)
#             forecast_index = pd.bdate_range(start = fit_end + pd.Timedelta(days = 1), periods = horizon)
#             forecast_mean = pd.Series(forecast.mean.iloc[-1].values, index = forecast_index)
#             forecast_std = pd.Series(np.sqrt(forecast.variance.iloc[-1].values), index = forecast_index)
#             mean = pd.concat([mean, forecast_mean])
#             std = pd.concat([std, forecast_std])
#             model_index = mean.index
#         else:
#             model_index = fit_index
#         upper = mean + k * std
#         lower = mean - k * std
#         plt.plot(model_index, mean, label = 'Fitted Mean', color = 'red', linestyle = '--')
#         plt.fill_between(model_index, lower, upper, color = 'red', alpha = 0.3, label = f'±{k} std band')
#     else:
#         plt.plot(index, series, label = 'Original Series', color = 'blue')
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_series(series: pd.Series, model = None, k = 2, horizon = 0):
    """
    Plot a time series along with a model fit and forecast (if provided).

    Parameters
    ----------
    series : pd.Series
        The complete time series data to plot. If a model fit is provided, the series may extend
        beyond its range.
    model : optional
        A fitted time series model (e.g., from `arch` or similar library). If provided, the function
        will plot the fitted mean and ±k standard deviation band from the model, and optionally 
        a forecast.
    k : int, default = 2
        Number of standard deviations for the confidence band.
    horizon : int, default = 0
        Number of periods to forecast beyond the end of the fit. If zero, no forecast is shown.
    """
    index = series.index
    plt.figure(figsize = (10, 5))
    if model:
        model_index = model.model.y.index
        model_start, model_end = model_index[0], model_index[-1]
        index_left = index[index <= model_start]
        index_mid = index[(index >= model_start) & (index <= model_end)]
        index_right = index[index >= model_end]
        label_used = False
        plt.plot(index_mid, series.loc[index_mid], label = 'Original Series (fit)', color = 'blue')
        if not index_left.empty:
            plt.plot(index_left, series.loc[index_left], label = 'Original Series (not fit)', color = "#B727D4")
            label_used = True
        if not index_right.empty:
            lbl = None if label_used else 'Original Series (not fit)'
            plt.plot(index_right, series.loc[index_right], label = lbl, color = "#B727D4")
        std = model.conditional_volatility
        mean = pd.Series(model.model.y, index = index_mid) - model.resid[index_mid]
        if horizon:
            forecast = model.forecast(horizon = horizon, start = model_end)
            forecast_index = pd.bdate_range(start = model_end + pd.Timedelta(days = 1), periods = horizon)
            forecast_mean = pd.Series(forecast.mean.iloc[-1].values, index = forecast_index)
            forecast_std = pd.Series(np.sqrt(forecast.variance.iloc[-1].values), index = forecast_index)
            mean = pd.concat([mean, forecast_mean])
            std = pd.concat([std, forecast_std])
            plot_index = mean.index
        else:
            plot_index = index_mid
        upper = mean + k * std
        lower = mean - k * std
        plt.plot(plot_index, mean, label = 'Fitted Mean', color = 'red', linestyle = '--', alpha = 0.6, linewidth = 1)
        plt.fill_between(plot_index, lower, upper, color = 'red', alpha = 0.3, label = f'±{k} std band')
    else:
        plt.plot(index, series, label = 'Original Series', color = 'blue')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(series: pd.Series, lags: int = 40):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    for a given time series.

    Parameters
    ----------
    series : pd.Series
        The complete time series to study the pairwise autocorrelation and partial
        autocorrelation structure.
    lags : int, default = 40
        The number of lags to include in the ACF and PACF plots.
    """
    _, axes = plt.subplots(1, 2, figsize = (12, 4))
    plot_acf(series.dropna(), ax = axes[0], lags = lags, title = 'ACF')
    plot_pacf(series.dropna(), ax = axes[1], lags = lags, title = 'PACF', method = 'ywm')
    plt.tight_layout()
    plt.show()