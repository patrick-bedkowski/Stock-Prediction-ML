import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def run_program():
    PREDICTION_DAYS = 60
    COMPANY = str(input('Type company name to view its future stock prices: '))
    
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)

    try:
        # get stock data
        data = web.DataReader(COMPANY, 'yahoo', start, end)
        print(data)
        train_model(data, PREDICTION_DAYS, COMPANY)
    except:
        print('Something went wrong')
        while True:
            response = input('Do you want to insert company name again? (Y/N): ')
            if response == 'Y':
                run_program()
                break
            elif response == 'N':
                print('Goodbye!')
                break
            else:
                print('Please insert proper option')

def train_model(data: DataFrame, PREDICTION_DAYS: int, COMPANY: str) -> None:
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

    # model.save

    # prepare test data

    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(COMPANY, 'yahoo', test_start, test_end)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # make predictions on test data

    x_test = []

    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # PLOT the test predictions

    plt.plot(actual_prices, color='black', label = 'actual')
    plt.plot(predicted_prices, color='green', label='predicted')
    plt.xlabel('TIME')
    plt.ylabel('share price')
    plt.legend()
    plt.savefig(f'{COMPANY}.png')

if __name__ == '__main__':
    run_program()
