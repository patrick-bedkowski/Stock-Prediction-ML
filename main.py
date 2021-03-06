import pandas_datareader as web
import datetime as dt
import sys

from trainer import train_model, predict, validate
from plotter import plot_stocks

def run_program():

    # if additional arguments have been passed
    if len(sys.argv) == 2:
        # get stock ticker symbol
        COMPANY = (sys.argv[1]).upper()
    elif len(sys.argv) == 1:
        COMPANY = str(input('Insert company stock ticker symbol: ')).upper()
    else:
        print('Invalid number of arguments have been passed.')
        sys.exit()
    
    # set parameters
    PREDICTION_DAYS = 30
    
    # learning data for the ML algorithm
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)

    try:
        # get stock data
        data = web.DataReader(COMPANY, 'yahoo', start, end)

        # train the model
        scaler, model = train_model(data, PREDICTION_DAYS, COMPANY) # , model
        
        # useful to validate the model
        # validate(data, PREDICTION_DAYS, COMPANY, scaler, model)
        
        # predict future values using the previously built model
        values_till_now, predicted_values, model_values_date = predict(PREDICTION_DAYS, COMPANY, scaler, model) # , model

        # plot stocks
        plot_stocks(values_till_now, predicted_values, model_values_date, COMPANY, PREDICTION_DAYS)
    except Exception as exc:
        print('Something went wrong', exc)
        while True:
            response = input('Do you want to insert company name again? (n/y): ')
            if response == 'y':
                run_program()
                break
            elif response == 'n':
                print('Goodbye!')
                break
            else:
                print('Please insert proper option')

if __name__ == '__main__':
    run_program()
