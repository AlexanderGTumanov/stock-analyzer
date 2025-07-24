import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

def fit_ARMA(series: pd.Series, p: int, q: int, summary: bool = False):
    model = ARIMA(series, order = (p, 0, q))
    arma = model.fit()
    
    if summary:
        print(arma.summary())
        print('\nModel Diagnostics:')
        print(f'  Log-Likelihood: {arma.llf:.2f}')
        print(f'  AIC: {arma.aic:.2f}')
        print(f'  BIC: {arma.bic:.2f}')

    return arma

def fit_ARCH(series: pd.Series, lags: int = 1, p: int = 1, q: int = 1):
    model = arch_model(
        series,
        mean = 'AR',
        lags = lags,
        vol = 'GARCH',
        p = p,
        q = q,
        dist = 'normal',
        rescale = False
    )
    arch = model.fit(disp = 'off', cov_type = 'robust')
    return arch

def print_diagnostics(model):
    print('Model Diagnostics:')
    print(f'  Log-Likelihood: {model.loglikelihood:.2f}')
    print(f'  AIC: {model.aic:.2f}')
    print(f'  BIC: {model.bic:.2f}')

def forecast(model, horizon):
    model_end = model.model.y.index[-1]
    forecast = model.forecast(horizon = horizon, start = model_end)
    index = pd.bdate_range(start = model_end + pd.Timedelta(days = 1), periods = horizon)
    mean = pd.Series(forecast.mean.iloc[-1].values, index = index)
    std = pd.Series(np.sqrt(forecast.variance.iloc[-1].values), index = index)
    return mean, std

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
            forecast_mean, forecast_std = forecast(model, horizon)
            mean = pd.concat([mean, forecast_mean])
            std = pd.concat([std, forecast_std])
            plot_index = mean.index
        else:
            plot_index = index_mid
        upper = mean + k * std
        lower = mean - k * std
        plt.plot(plot_index, mean, label = 'ARMA Forecast Mean', color = 'red', linestyle = '--', alpha = 0.6, linewidth = 1)
        plt.fill_between(plot_index, lower, upper, color = 'red', alpha = 0.3, label = f'±{k} std band')
        plt.title('ARCH fit')
    else:
        plt.plot(index, series, label = 'Original Series', color = 'blue')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()