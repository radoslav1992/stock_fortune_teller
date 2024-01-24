import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pmdarima as pm
from datetime import date, timedelta

# Example: List of tickers
ticker_list = {'AAPL': 'Apple Inc', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc', 'AMZN' : 'Amazon', 'TSLA' : 'Tesla', 'FB' : 'Facebook', 'NFLX' : 'Netflix'}

def main():
    st.title('Hybrid Stock Price Forecasting App (ARIMA and Random Forest)')


    # Convert the ticker list into a format suitable for selection
    formatted_ticker_list = [f"{ticker} - {name}" for ticker, name in ticker_list.items()]

    # Ticker selection with a search-enabled dropdown
    selected_ticker = st.selectbox('Select Stock Ticker', formatted_ticker_list)
    ticker = selected_ticker.split(' - ')[0]  # Extract the ticker symbol

    # Date picker for the range
    start_date = st.date_input('Start Date', date(2020, 1, 1))
    end_date = st.date_input('End Date', date.today())

    if start_date < end_date:
        # Fetch stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if not stock_data.empty:
            stock_data.reset_index(inplace=True)
            stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days

            # Fit ARIMA model
            arima_model = pm.auto_arima(stock_data['Close'], seasonal=True, m=5, suppress_warnings=True)
            arima_pred = arima_model.predict_in_sample()

            # Calculate residuals
            residuals = stock_data['Close'] - arima_pred

            # Fit Random Forest on residuals
            X = stock_data['Days'].values.reshape(-1, 1)  # Feature: time as integer
            y = residuals.values  # Target: residuals
            rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
            rf_model.fit(X, y)

            # Make predictions
            forecast_period = (pd.to_datetime(end_date) - stock_data['Date'].max()).days
            future_days = stock_data['Days'].max() + np.arange(1, forecast_period + 1)
            future_dates = stock_data['Date'].max() + pd.to_timedelta(np.arange(1, forecast_period + 1), 'D')
            arima_forecast = arima_model.predict(n_periods=forecast_period)
            rf_forecast = rf_model.predict(future_days.reshape(-1, 1))

            # Combine forecasts
            hybrid_forecast = arima_forecast + rf_forecast

            # Plotting the results
            plt.figure(figsize=(10, 5))
            plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Prices', color='blue')
            plt.plot(stock_data['Date'], arima_pred, label='ARIMA Predictions', color='green')
            plt.plot(future_dates, arima_forecast, label='ARIMA Forecast', color='green', linestyle='dashed')
            plt.plot(future_dates, hybrid_forecast, label='Hybrid Forecast', color='red', linestyle='dashed')
            plt.fill_between(future_dates,
                             (arima_forecast - rf_model.estimators_[0].predict(future_days.reshape(-1, 1))),
                             (arima_forecast + rf_model.estimators_[0].predict(future_days.reshape(-1, 1))),
                             color='pink', alpha=0.3, label='Forecast Confidence Interval')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Stock Price Forecast for {ticker}')
            plt.legend()
            st.pyplot(plt)
        else:
            st.error('No data available for the selected ticker.')
    else:
        st.error('End Date must be after Start Date. Please select a valid date range.')


if __name__ == '__main__':
    main()