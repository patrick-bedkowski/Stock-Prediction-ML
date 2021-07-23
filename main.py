import pandas_datareader as web
import datetime as dt

from trainer import train_model, predict

def run_program():
    PREDICTION_DAYS = 60
    COMPANY = str(input('Type company name to view its future stock prices: '))
    
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime.now()

    try:
        # get stock data
        data = web.DataReader(COMPANY, 'yahoo', start, end)
        scaler, model = train_model(data, PREDICTION_DAYS, COMPANY)
        predict(data, PREDICTION_DAYS, COMPANY, scaler, model)
    except Exception as s:
        print(s)
        print('Something went wrong')
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
