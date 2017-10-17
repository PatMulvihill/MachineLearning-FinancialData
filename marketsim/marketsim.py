"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    file1 = pd.read_csv(orders_file, index_col='Date',parse_dates=True, na_values=['nan'])
    file1 = file1.sort_index()
    symbols = file1['Symbol'].values
    symbols = list(set(symbols))
    prices = get_data(symbols, pd.date_range(file1.index[0], file1.index[-1]))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    new_dataframe = pd.DataFrame(np.zeros((prices.shape[0], prices.shape[1])), index=prices.index, columns=prices.columns)
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
        value = m*each[1]['Shares']*prices.ix[each[0]:,each[1]['Symbol']] - com - impact_val * each[1]['Shares']*prices.ix[each[0]:,each[1]['Symbol']]
        new_dataframe.ix[each[0]:, 'Cash'] -= value[each[0]]
        new_dataframe.ix[each[0]:,each[1]['Symbol']] += value
    portvals = new_dataframe.sum(axis=1).to_frame()
    return portvals

def author():
    return 'lwang496'

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
