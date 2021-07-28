from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt

from plotter import plot_validate

# ML
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


def train_model(data: DataFrame, PREDICTION_DAYS: int, COMPANY: str) -> None:
    """
    Parameters:
        data: datareader of stocks
        PREDICTION_DAYS: days to look previously,
            before choosing what the value is of the next day
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

    for x in range(2*PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-2*PREDICTION_DAYS:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # build the model
    model = Sequential()

    # units - 50
    model.add(LSTM(units = 40, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 40, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 40))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1)) # prediction of the next closing price

    # compile model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, epochs = 25, batch_size = 25)

    model.save(f'model', include_optimizer=True)  # {COMPANY}_{PREDICTION_DAYS}

    return scaler, model
    

def predict(PREDICTION_DAYS: int, COMPANY: str, scaler, model) -> None:
    """
    Parameters:
        PREDICTION_DAYS: days to look previously,
            before choosing what the value is of the next day
        COMPANY: name of the company
        scaler: scaler from ML algorithm
        model: ML model of the reviewed company
    Operations:
        Function predicts stock value of specific company
        number of days into the future. 
    Returns:
        None
    """

    # load saved the model from disk
    # model = load_model('model')

    # load previous stock data to feed the model
    test_start = dt.datetime.now() - dt.timedelta(5*PREDICTION_DAYS)
    test_end = dt.datetime.now()

    print('Timeframe: ', test_start, test_end)

    test_data = web.DataReader(COMPANY, 'yahoo', test_start, test_end)
    
    # get values from previous stocks
    model_values = test_data['Close'].values
    # save date time of the values, convert them to string
    model_values_date = test_data.index.strftime('%d-%m-%Y').tolist()

    values_till_now = model_values

    # reshape data
    model_inputs = model_values.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    predicted_values = []

    # iterate through list
    for x in list(range(0, PREDICTION_DAYS))[::-1]:

        # data that is fed into a model
        # list of values days before the predicted one
        data = [model_inputs[len(model_inputs) - x - 2*PREDICTION_DAYS:len(model_inputs) - x, 0]]

        # format data
        data = np.array(data)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))

        # predict next price
        predicted_prices = model.predict(data)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # append predicted price to the value list
        ##model_values = np.append(model_values, predicted_prices)
        predicted_values.append(predicted_prices.item(0))

        # this part was an early concept that assumed feeding
        # the predicted values into a model to predict another value
        # but inserting false data could influence the output significantly

        # update model inputs
        ##model_inputs = model_values.reshape(-1, 1)
        ##model_inputs = scaler.transform(model_inputs)
    
    return values_till_now, predicted_values, model_values_date


def validate(data: DataFrame, PREDICTION_DAYS: int, COMPANY: str, scaler, model) -> None:
    """
    Parameters:
        data: datareader of stocks
        PREDICTION_DAYS: days to look previously,
            before choosing what the value is of the next day
        COMPANY: name of the company
        scaler: scaler from ML algorithm
        model: ML model of the reviewed company
    Operations:
        Function validates the correctness of the model by comparing
        predicted data and real data from the past
    Returns:
        None
    """
    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(COMPANY, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(2*PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - 2*PREDICTION_DAYS:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plot_validate(actual_prices, predicted_prices)
