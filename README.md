# Stock Analyzer

A simple Python project to analyze stock data and forecast volatility using an ARCH model.  
This project is designed as a small portfolio piece to demonstrate Python skills in data analysis and basic time-series modeling.

---

## What It Does

- Downloads historical stock prices (e.g., from Yahoo Finance).
- Computes daily returns and log returns.
- Fits a simple ARCH model to estimate volatility and produce short-term volatility forecasts.
- Builds and trains a PyTorch NN with the same functionality.
- Plots and compares the predictions of the two models and accuracies of their forecasts.
- A notebook file is provided, in which this workflow is implemented to model the behavior of three stocks in the aftermath of high-volatility events.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/AlexanderGTumanov/stock-analyzer>
   cd <stock-analyzer>

---

## Contents of the Notebook

Notebook `/notebooks/stock_analyzer.ipynb` 

---

## Contents of the `/src` folder

The `/src` folder contains three modules: `analysis.py`, `model_arch.py`, `model_pytorch.py`.

### `analysis.py`

This file contains tools for general time series analysis:

- **retrieve_stock(ticker: str, start, end)**: Retrieves the adjusted close price of the stock identified by **ticker** between the dates **start** and **end** using *yfinance*, and returns the data as a *pandas* Series.

- **log_returns(series: pd.Series)**: Computes the logarithmic returns of **series**.

- **rolling_mean(series: pd.Series, window: int = None)**: Smooths **series** using a rolling mean. The window length is controlled by **window**; if not provided, it defaults to 1% of the total dataset length.

- **stationarity_test(series: pd.Series)**: Performs stationarity tests on a time series using the *Augmented Dickey-Fuller (ADF)* and *Kwiatkowski-Phillips-Schmidt-Shin (KPSS)* tests. Prints the test statistic, p-value, and critical values for both tests, and provides a basic assessment of stationarity based on a 5% significance level.

- **plot_series(series: pd.Series)**: Plots **series**.

- **plot_acf_pacf(series: pd.Series, lags: int = 40)**: Plots the *autocorrelation function* and *partial autocorrelation function* of **series** up to the number of lags specified by **lags**.

- **compare_gaussian_nll(series: pd.Series, mean1: pd.Series, std1: pd.Series, mean2: pd.Series, std2: pd.Series, label1: str = "NN", label2: str = "ARCH")**: The main tool for model comparisons. Compares the Gaussian negative log-likelihood (NLL) for two forecasts over the overlapping portion of the time series. **series** is the observed time series against which both forecasts are evaluated; **mean1** and **std1** are the mean and standard deviation forecasts of the first model, while **mean2** and **std2** belong to the second model. By default, the function assumes the first forecast comes from the NN and the second from the ARCH model, labeling them accordingly. Labels can be customized via **label1** and **label2**.

### `model_arch.py`

This file contains specialized tools for ARMA/ARCH model fits:
