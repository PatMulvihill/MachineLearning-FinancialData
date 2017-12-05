"""Author: Lu Wang, lwang496, lwang496@gatech.edu"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

# Author: Lu Wang lwang496

def compute_portvals(file1, start_val=1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here


    symbols = file1['Symbol'].values
    symbols = list(set(symbols))
    prices = get_data(symbols, pd.date_range(file1.index[0], file1.index[-1]))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    # construct a new dataframe
    new_dataframe = pd.DataFrame(np.zeros((prices.shape[0], prices.shape[1])), index=prices.index,
                                 columns=prices.columns)
    new_dataframe['Cash'] = start_val
    for each in file1.iterrows():
        com = 0
        impact_val = 0

        if each[1]['Order'] == 'BUY':
            m = 1
            com = commission
            impact_val = impact

        if each[1]['Order'] == 'SELL':
            m = -1
            com = commission
            impact_val = impact
        # caculate the value shares*prices
        value = m * each[1]['Shares'] * prices.ix[each[0]:, each[1]['Symbol']]
        # deduct commision and impact
        total_value = value[each[0]] + com + impact_val * abs(value[each[0]])
        new_dataframe.ix[each[0]:, 'Cash'] -= total_value
        new_dataframe.ix[each[0]:, each[1]['Symbol']] += value

    portvals = new_dataframe.sum(axis=1).to_frame()

    return portvals


def author():
    return 'lwang496'

