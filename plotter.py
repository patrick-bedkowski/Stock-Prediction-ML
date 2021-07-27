from typing import List
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime as dt
from pandas import Timestamp

import pandas as pd
from tensorflow.python.keras.backend import dtype_numpy

def plot_validate(actual_prices, predicted_prices) -> None:
    # plot
    plt.plot(actual_prices, color="black")
    plt.plot(predicted_prices, color="green")
    plt.savefig('hello.png')

def plot_stocks(x_values, predicted_now: List[int], model_values_date, COMPANY, PREDICTION_DAYS) -> None:
    """
    Plots predicted stock value with previous values.
    """
    x_values = x_values.tolist()

    # modify the values so the predicted ones are exactly 1/3 of (x_values+1)
    x_values = x_values[13:]

    model_values_date = model_values_date[13:len(model_values_date)-10]

    # create fig
    fig, ax = plt.subplots(figsize=(16,10))

    # prepare data
    new_x = [None]*(len(x_values)-1)
    new_x.append(x_values[-1])
    new_x.extend(predicted_now)

    # predicted
    ax.plot(new_x, 'o', color = '#FFB32A')
    ax.plot(new_x, color = '#FFB93A', label = 'Predicted value') 

    ax.plot(x_values, 'o', color='#29526D')
    ax.plot(x_values, color = '#8DB0C7', label = 'Previous values')

    ax.legend()
    plt.legend(loc=2, prop={'size': 16})

    merged_x_data = list(x_values + predicted_now)

    # find min max value of stocks
    max_y = math.ceil(max(merged_x_data))
    min_y = math.floor(min(merged_x_data))
    # set ytics
    ax.set_yticks(np.arange(min_y-0.5, max_y+1, 0.5))

    # set xtics 
    x_tics = np.arange(0, len(x_values)+len(predicted_now)+1, step=10)
    ax.set_xticks(x_tics)

    model_values_date = model_values_date[::10]

    x_labels = []
    for day in model_values_date:
        x_labels.append(day)

    # today
    # DATE X LABELS
    today = dt.date.today()
    d1 = today.strftime("%d-%m-%Y")
    
    # append today
    x_labels.append(d1)
    
    #print(x_labels)
    #print(type(x_labels))

    start = today
    end = today + dt.timedelta(len(predicted_now)+1)
    date_generated = [start + dt.timedelta(days=x) for x in range(0, (end-start).days)]

    predicted_labels = []
    for date in date_generated:
        predicted_labels.append(date.strftime("%d/%m/%Y"))
    
    predicted_labels = predicted_labels[10::10]

    # append predicted labels
    x_labels.extend(predicted_labels)
    
    # set xtics labels to the days
    labels_previous = list(range(-(len(x_values)-1), -9, 10))
    labels_predicted = [d1] + list(range(10, len(predicted_now)+1, 10))

    # x_tics_labels = np.concatenate(labels_previous, labels_predicted)
    ax.set_xticklabels(x_labels)
    
    # rotate x labels
    plt.xticks(rotation=10)

    # set x, y labels padding 
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)
    
    ax.set_title(f'Previous and predicted stock values of {COMPANY}', size = 16)
    ax.set_xlabel('Days', size = 16)
    ax.set_ylabel('Stock Value', size = 16)

    plt.grid(color='grey', linestyle='-', linewidth=0.25)

    plt.box(False)  # borderless

    #print(len(predicted_now))
    #print(len(x_values))

    plt.savefig(f"stock_{COMPANY}.png", dpi=150, bbox_inches='tight')