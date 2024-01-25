import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
import pmdarima as pm
from datetime import date
from streamlit_option_menu import option_menu

# Constants
TICKER_LIST = {
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'FB': 'Facebook',
    'NFLX': 'Netflix',
    'NVDA': 'Nvidia',
    'Custom': 'Custom Ticker'
}

# Page Configurations
# Streamlit UI enhancements
st.set_page_config(page_title="Hybrid Stock Price Forecasting", layout="wide")
st.markdown("""
    <style>
    .reportview-container {
        background: #FFFFFF;
        color: #111111;
    }
    .sidebar .sidebar-content {
        background: #F0F2F6;
    }
    h1 {
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

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
    # Enhanced Navigation Bar
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "Forecast", "About"],
                               icons=['house', 'graph-up', 'info-circle'],
                               menu_icon="cast", default_index=0)

    if selected == "Home":
        st.title('Welcome to the Hybrid Stock Price Forecasting App')
        st.write("Navigate to the Forecast section to start.")

    elif selected == "Forecast":
        st.title('Hybrid Stock Price Forecasting App')

        # Ticker selection with improved layout
        st.sidebar.header("Settings")
        selected_ticker = st.sidebar.selectbox('Select Stock Ticker', [f"{ticker} - {name}" for ticker, name in TICKER_LIST.items()])

        # Conditional text input for custom ticker
        if selected_ticker == 'Custom - Custom Ticker':
            custom_ticker = st.sidebar.text_input('Enter your custom ticker')
            if custom_ticker:
                selected_ticker = custom_ticker.upper()
        ticker = selected_ticker.split(' - ')[0]

        # Date range selection with enhanced sidebar
        start_date = st.sidebar.date_input('Start Date', date(2020, 1, 1))
        end_date = st.sidebar.date_input('End Date', date.today())

        # Main content layout
        if start_date < end_date and ticker:
            stock_data = download_stock_data(ticker, start_date, end_date)
            if not stock_data.empty:
                prepare_stock_data(stock_data)
                display_stock_forecast(stock_data, end_date)
            else:
                st.error('No data available for the selected ticker.')
        else:
            st.error('Please select a valid date range and ticker.')

    elif selected == "About":
        st.title("About this App")
        st.write("This app provides a hybrid stock price forecast using ARIMA and Random Forest models. Developed with Streamlit.")

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
    """Plot the stock forecast based on ARIMA and Random Forest models using Plotly."""
    forecast_period = calculate_forecast_period(data, end_date)
    last_actual_date = data['Date'].iloc[-1]
    future_dates = calculate_future_dates(last_actual_date, forecast_period)

    future_days = np.array([(date - data['Date'].min()).days for date in future_dates])
    future_days_reshaped = future_days.reshape(-1, 1)

    arima_forecast = arima_model.predict(n_periods=forecast_period)
    rf_forecast = rf_model.predict(future_days_reshaped)

    hybrid_forecast = arima_forecast + rf_forecast

    plot_forecast_graph(data, future_dates, arima_forecast, hybrid_forecast)

def calculate_forecast_period(data, end_date):
    """Calculate the forecast period in days."""
    return (pd.to_datetime(end_date) - data['Date'].max()).days

def calculate_future_dates(last_actual_date, forecast_period):
    """Generate future dates for forecasting."""
    # Generate a date range starting the day after the last actual date
    return pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=forecast_period, freq='D')


def plot_forecast_graph(data, future_dates, arima_forecast, hybrid_forecast):
    """Plot the actual prices, ARIMA forecast, and Hybrid forecast using Plotly."""
    fig = go.Figure()

    # Actual Prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Prices'))

    # ARIMA Predictions
    fig.add_trace(go.Scatter(x=future_dates, y=arima_forecast, mode='lines', name='ARIMA Predictions', line=dict(dash='dot')))

    # Hybrid Forecast
    fig.add_trace(go.Scatter(x=future_dates, y=hybrid_forecast, mode='lines', name='Hybrid Forecast', line=dict(dash='dot')))

    fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
