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

def fit_ARCH(series: pd.Series, lags: int = 1, p: int = 1, q: int = 1, summary: bool = False):
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

    if summary:
        print(arch.summary())
        print('\nModel Diagnostics:')
        print(f'  Log-Likelihood: {arch.loglikelihood:.2f}')
        print(f'  AIC: {arch.aic:.2f}')
        print(f'  BIC: {arch.bic:.2f}\n')

    return arch