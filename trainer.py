from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import os

from plotter import plot_validate

# ML
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


def train_model(data: DataFrame, PREDICTION_DAYS: int, COMPANY: str, EPOCHS: int):
    """
    Trains a stock price prediction model.

    Parameters:
        data: Stock price data
        PREDICTION_DAYS: Number of previous days to consider
        COMPANY: Stock ticker
        EPOCHS: Number of training epochs

    Returns:
        Scaler and trained model
    """
    print("🚀 Starting training...")

    # Scale data
    scaler = MinMaxScaler(feature_range=(data['Close'].min(), data['Close'].max()))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare training data
    x_train, y_train = [], []
    for x in range(2 * PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x - 2 * PREDICTION_DAYS:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(units=20, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=20, return_sequences=True),
        Dropout(0.2),
        LSTM(units=20),
        Dropout(0.2),
        Dense(units=1)
    ])

    # Compile and train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=6)

    # Save model
    model.save(f'model_{EPOCHS}.h5')
    print(f"✅ Model saved as model_{EPOCHS}.h5")

    return scaler, model


def load_existing_model(EPOCHS: int):
    """
    Loads a saved model and initializes a scaler.

    Parameters:
        EPOCHS: The number of epochs the saved model was trained on

    Returns:
        Scaler and loaded model
    """
    if not os.path.exists(f'model_{EPOCHS}.h5'):
        raise FileNotFoundError(f"❌ Model file 'model_{EPOCHS}.h5' not found!")

    print(f"📂 Loading model from 'model_{EPOCHS}.h5'...")
    model = load_model(f'model_{EPOCHS}.h5')

    # Dummy scaler to be compatible with training
    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler, model


def predict(PREDICTION_DAYS: int, COMPANY: str, scaler, model):
    """
    Predicts future stock prices.

    Parameters:
        PREDICTION_DAYS: Number of future days to predict
        COMPANY: Stock ticker symbol
        scaler: Scaler used for normalizing data
        model: Trained ML model

    Returns:
        Tuple of actual values, predicted values, and corresponding dates
    """
    print(f"🔮 Predicting stock prices for {COMPANY}...")

    # Define timeframe for historical data
    test_start = dt.datetime.now() - dt.timedelta(days=5 * PREDICTION_DAYS)
    test_end = dt.datetime.now()

    test_data = yf.download(COMPANY, start=test_start, end=test_end)
    if test_data.empty:
        print(f"⚠️ Warning: No data returned for {COMPANY}.")
        return None, None, None

    model_values = test_data['Close'].values
    model_values_date = test_data.index.strftime('%d-%m-%Y').tolist()
    values_till_now = model_values

    # Scale and reshape input data
    model_inputs = model_values.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    predicted_values = []

    # Generate predictions
    for x in list(range(0, PREDICTION_DAYS))[::-1]:
        data = [model_inputs[len(model_inputs) - x - 2 * PREDICTION_DAYS: len(model_inputs) - x, 0]]
        data = np.array(data, dtype=np.float32)
        data = data.reshape((data.shape[0], data.shape[1], 1))

        predicted_prices = model.predict(data)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        predicted_values.append(predicted_prices.item(0))

    return values_till_now, predicted_values, model_values_date
