import pandas_datareader as web
import datetime as dt
import os
import sys
import yfinance as yf
from trainer import train_model, predict, load_existing_model
from plotter import plot_stocks

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        # Check if data is empty
        if data.empty:
            print(f"Warning: No data returned for {ticker}. Check the ticker symbol or try again later.")
            return None

        print(f"Successfully fetched data for {ticker}")
        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def run_program():
    COMPANY = "TSLA"
    PREDICTION_DAYS = 10
    EPOCHS = 10

    # 🔥 New Variable: Load Model from Checkpoint
    LOAD_FROM_CHECKPOINT = False  # Change to False to retrain model

    start = dt.datetime(2024, 6, 1)
    end = dt.datetime(2025, 3, 3)

    data = fetch_stock_data(COMPANY, start, end)
    if data is None:
        print("No stock data available. Exiting...")
        return

    # 🔥 Load model from checkpoint or train a new one
    if LOAD_FROM_CHECKPOINT and os.path.exists(f'model_{EPOCHS}.h5'):
        print("✅ Loading model from checkpoint...")
        scaler, model = load_existing_model(EPOCHS)
    else:
        print("🚀 Training a new model...")
        scaler, model = train_model(data, PREDICTION_DAYS, COMPANY, EPOCHS)

    # Predict future values
    values_till_now, predicted_values, model_values_date = predict(PREDICTION_DAYS, COMPANY, scaler, model)

    # Plot stock prices
    plot_stocks(values_till_now, predicted_values, model_values_date, COMPANY, PREDICTION_DAYS)

if __name__ == '__main__':
    run_program()
