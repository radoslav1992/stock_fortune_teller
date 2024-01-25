import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pmdarima as pm
from datetime import date

# Constants
TICKER_LIST = {
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'FB': 'Facebook',
    'NFLX': 'Netflix'
}

@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    """Download and return stock data for a given ticker and date range."""
    data = yf.download(ticker, start=start_date, end=end_date)
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data.reset_index(inplace=True)
    return data

@st.cache_data
def train_random_forest_model(features, targets):
    """Train a Random Forest model on the given features and targets."""
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(features, targets)
    return model

def train_arima_model(series):
    """Train an ARIMA model on the given time series."""
    return pm.auto_arima(series, seasonal=True, m=5, suppress_warnings=True)

def main():
    """Main function to run the Streamlit app."""
    st.title('Hybrid Stock Price Forecasting App')

    # Ticker selection
    formatted_tickers = [f"{ticker} - {name}" for ticker, name in TICKER_LIST.items()]
    selected_ticker = st.selectbox('Select Stock Ticker', formatted_tickers)
    ticker = selected_ticker.split(' - ')[0]

    # Date range selection
    start_date = st.date_input('Start Date', date(2020, 1, 1))
    end_date = st.date_input('End Date', date.today())

    if start_date < end_date and ticker:
        stock_data = download_stock_data(ticker, start_date, end_date)

        if not stock_data.empty:
            prepare_stock_data(stock_data)
            display_stock_forecast(stock_data, end_date)
        else:
            st.error('No data available for the selected ticker.')
    else:
        st.error('Please select a valid date range and ticker.')

def prepare_stock_data(data):
    """Prepare stock data for modeling."""
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days

def display_stock_forecast(data, end_date):
    """Display the stock forecast using ARIMA and Random Forest."""
    arima_model = train_arima_model(data['Close'])
    arima_predictions = arima_model.predict_in_sample()

    residuals = data['Close'] - arima_predictions
    features = data['Days'].values.reshape(-1, 1)
    targets = residuals.values

    rf_model = train_random_forest_model(features, targets)
    plot_stock_forecast(data, arima_model, rf_model, end_date)


def plot_stock_forecast(data, arima_model, rf_model, end_date):
    """Plot the stock forecast based on ARIMA and Random Forest models."""
    forecast_period = calculate_forecast_period(data, end_date)
    last_actual_date = data['Date'].iloc[-1]
    future_dates = calculate_future_dates(last_actual_date, forecast_period)

    # Convert future_dates to a NumPy array of days since the start for prediction
    future_days = np.array([(date - data['Date'].min()).days for date in future_dates])
    future_days_reshaped = future_days.reshape(-1, 1)  # Reshaping to 2D array

    # Generate ARIMA and RF forecasts
    arima_forecast = arima_model.predict(n_periods=forecast_period)
    rf_forecast = rf_model.predict(future_days_reshaped)  # Using the reshaped array

    # Combine ARIMA and RF forecasts for the hybrid forecast
    hybrid_forecast = arima_forecast + rf_forecast

    # Now plot the forecasts alongside the actual prices
    plot_forecast_graph(data, future_dates, arima_forecast, hybrid_forecast)

def calculate_forecast_period(data, end_date):
    """Calculate the forecast period in days."""
    return (pd.to_datetime(end_date) - data['Date'].max()).days

def calculate_future_dates(last_actual_date, forecast_period):
    """Generate future dates for forecasting."""
    # Generate a date range starting the day after the last actual date
    return pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=forecast_period, freq='D')


def plot_forecast_graph(data, future_dates, arima_forecast, hybrid_forecast):
    """Plot the actual prices, ARIMA forecast, and Hybrid forecast."""
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Actual Prices', color='blue')

    # Ensure that future_dates starts right after the last date in the actual data
    last_actual_date = data['Date'].iloc[-1]
    future_dates = calculate_future_dates(last_actual_date, len(arima_forecast))

    # Plot the ARIMA forecast and the hybrid forecast using the correct dates
    plt.plot(future_dates, arima_forecast, label='ARIMA Predictions', color='green', linestyle='dashed')
    plt.plot(future_dates, hybrid_forecast, label='Hybrid Forecast', color='red', linestyle='dashed')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Forecast')
    plt.legend()
    st.pyplot(plt)

if __name__ == '__main__':
    main()
