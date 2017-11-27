import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
import marketsimcode as mk

start_date_train = dt.datetime(2008,1,01)
end_date_train = dt.datetime(2009,12,31)
start_date_test = dt.datetime(2010,1,01)
end_date_test = dt.datetime(2011,12,31)

def author():
    return 'lwang496'

def testPolicy( symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
    lookback = 21
    price = get_data([symbol], pd.date_range(sd,ed))
    price.fillna(method='ffill', inplace=True)
    price.fillna(method='bfill', inplace=True)
    sma = price.rolling(window = lookback, min_periods = lookback).mean()
    sma.fillna(method='ffill', inplace=True)
    sma.fillna(method='bfill', inplace=True)

    # caculate bbp
    rolling_std = price.rolling(window = lookback, min_periods = lookback).std()
    top_band = sma + (2*rolling_std)
    bottom_band = sma - (2*rolling_std)
    bbp = (price - bottom_band) / (top_band- bottom_band)
    # turn sma into price/sma ratio
    sma_n = price/sma

    #caculate momentum
    momentum = (price / price.copy().shift(lookback))-1

    # orders
    orders = price.copy()
    orders.ix[:,:] = np.NaN
    # create a binary array showing when price is above sma
    sma_cross = pd.DataFrame(0,index = sma_n.index, columns = sma_n.columns)
    sma_cross[sma_n >= 1] =1
    #turn that array into that only shows the crossings
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0
    # caculate the result of the overall strategy
    del sma_n['SPY']
    del bbp['SPY']
    del orders['SPY']
    del momentum['JPM']
    orders = pd.concat([orders, bbp, sma_n, momentum], axis=1)

    orders.fillna(method='ffill', inplace=True)
    orders.fillna(method='bfill', inplace=True)
    orders.columns = ['orders', 'bbp', 'sma', 'momentum']
    orders['Position'] = np.nan
    orders['BUY'] = np.nan
    orders['SELL'] =np.nan
    if orders['sma'].ix[0] > 1.05:
        orders['Position'].ix[0] = -1000
        orders['SELL'].ix[0] = 1000
    elif orders['sma'].ix[0] < 0.95:
        orders['Position'].ix[0] = 1000
        orders['BUY'].ix[0] = 1000
    else:
        orders['Position'].ix[0] = 0

    for i in range(1, orders.shape[0] - 1):
        if orders['sma'].ix[i] > 1.05 and orders['bbp'].ix[i]>1:
            if orders['Position'].ix[i - 1] == 1000:
                orders['Position'].ix[i] = -1000
                orders['SELL'].ix[i] = 2000
            elif orders['Position'].ix[i - 1] == 0:
                orders['Position'].ix[i] = -1000
                orders['SELL'].ix[i] = 1000
            else:
                orders['Position'].ix[i] = orders['Position'].ix[i - 1]

        elif orders['sma'].ix[i] < 0.95 and orders['bbp'].ix[i]<0:

            if orders['Position'].ix[i - 1] == -1000:
                orders['Position'].ix[i] = 1000
                orders['BUY'].ix[i] = 2000

            elif orders['Position'].ix[i - 1] == 0:
                orders['Position'].ix[i] = 1000
                orders['BUY'].ix[i] = 1000
            else:
                orders['Position'].ix[i] = orders['Position'].ix[i - 1]
        else:
            orders['Position'].ix[i] = orders['Position'].ix[i - 1]

    order_list = []
    for day in orders.index:

        if orders.ix[day, 'BUY'] == 1000:
            order_list.append([day.date(), symbol, 'BUY', 1000])
        elif orders.ix[day, 'BUY'] == 2000:
            order_list.append([day.date(), symbol, 'BUY', 2000])
        elif orders.ix[day, 'SELL'] == 2000:
            order_list.append([day.date(), symbol, 'SELL', 2000])
        elif orders.ix[day, 'SELL'] == 1000:
            order_list.append([day.date(), symbol, 'SELL', 1000])

    order_list.append([ed.date(), symbol, 'SELL', 0])
    order_list_final = pd.DataFrame(np.array(order_list), columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list_final.set_index('Date', inplace=True)
    return order_list_final









