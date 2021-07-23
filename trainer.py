from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import pandas_datareader as web

# PLOT
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


def train_model(data: DataFrame, PREDICTION_DAYS: int, COMPANY: str) -> None:
    """
    Parameters:
        data: datareader of stocks
        PREDICTION_DAYS: how many days ahead to predict
        COMPANY: name of the company
    Returns:
        Scaler and model
    """
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    
    # prepare train data
    x_train = []
    y_train = []

    for x in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-PREDICTION_DAYS:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # build the model
    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1)) # prediction of the next closing price

    # compile model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, epochs = 25, batch_size = 32)

    # model.save('FB', include_optimizer=True)
    return scaler, model
  

def predict(data: DataFrame, PREDICTION_DAYS: int, COMPANY: str, scaler, model) -> None:

    # load the model from disk
    # model = load_model('FB')

    # stock close values from {test_start} 
    model_inputs = data['Close'][len(data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # # Predict next days data
    real_data = [model_inputs[len(model_inputs) + 1 - PREDICTION_DAYS: len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    stocks_till_now = scaler.inverse_transform(real_data[-1])

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    
    print(f"Predicted closing price for tomorrow: {int(prediction)}")

    plot_stocks(stocks_till_now, int(prediction), COMPANY)

def plot_stocks(x_values, predicted_now, COMPANY) -> None:
    """
    Plots predicted stock value with previous values.
    """
    
    fig, ax = plt.subplots(figsize=(16,10))

    new_x = [None]*(len(x_values)-1)
    new_x.append(x_values[-1])
    new_x.append(predicted_now)

    # predicted
    ax.plot(new_x, 'o', color = '#F49948')
    ax.plot(new_x, color = '#F49948', label = 'Predicted value')

    # previous values
    ax.plot(x_values, 'o', color='#29526D')
    ax.plot(x_values, color = '#8DB0C7', label = 'Previous values')

    ax.legend()
    plt.legend(loc=2, prop={'size': 16})

    labels = [70, 60, 50, 40, 30, 20, 10, 0]
    ax.set_xticklabels(labels)
    
    ax.set_title(f'Previous and predicted stock values of {COMPANY}', size = 16)
    ax.set_xlabel('Days', size = 16)
    ax.set_ylabel('Stock Value', size = 16)

    plt.grid(color='grey', linestyle='-', linewidth=0.25)

    plt.box(False)  # borderless

    plt.savefig(f"stock_{COMPANY}.png", dpi=100, bbox_inches='tight')
