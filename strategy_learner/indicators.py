"""Author: Lu Wang, lwang496, lwang496@gatech.edu"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data


start_date_train = '2008-01-01'
end_date_train = '2009-12-31'
start_date_test = '2010-01-01'
end_date_test = '2011-12-31'

# Author: Lu Wang lwang496

def author():
    return 'lwang496'

def caculate( symbols, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    lookback = 21
    price = get_data([symbols], pd.date_range(sd,ed))
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



    # create a binary array showing when price is above sma
    sma_cross = pd.DataFrame(0,index = sma_n.index, columns = sma_n.columns)
    sma_cross[sma_n >= 1] =1
    #turn that array into that only shows the crossings
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0



    sma_n.fillna(method='ffill', inplace=True)
    sma_n.fillna(method='bfill', inplace=True)
    bbp.fillna(method='ffill', inplace=True)
    bbp.fillna(method='bfill', inplace=True)
    momentum.fillna(method='ffill', inplace=True)
    momentum.fillna(method='bfill', inplace=True)
    top_band.fillna(method='ffill', inplace=True)
    top_band.fillna(method='bfill', inplace=True)
    bottom_band.fillna(method='ffill', inplace=True)
    bottom_band.fillna(method='bfill', inplace=True)



    return sma, sma_n, bbp, top_band, bottom_band, momentum

