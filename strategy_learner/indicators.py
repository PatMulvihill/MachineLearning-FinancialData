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

symbols = 'JPM'
price = get_data([symbols], pd.date_range(start_date_train,end_date_test))
del price['SPY']
price = price/price.ix[0]


sma, sma_n, bbp, top_band, bottom_band, momentum = caculate( symbols, sd=dt.datetime(2008,1,1), ed=dt.datetime(2011,12,31))

print sma, sma_n, bbp

# Construct sma frame
del sma['SPY']
sma = sma/sma.ix[0]


del sma_n['SPY']
sma_n = sma_n/sma_n.ix[0]


sma_frame = pd.concat([price, sma, sma_n], axis=1)
sma_frame.columns = ['Price', 'SMA', 'Price/SMA']

img1= sma_frame.plot( fontsize=12,
                  title='Indicator1 Price/SMA')
img1.set_xlabel("Date")
img1.set_ylabel("Normalized Price")


# Construct momentum frame
del momentum['SPY']

mom_frame = pd.concat([price, momentum], axis=1)
mom_frame.columns = ['Price', 'momentum']


img2= mom_frame.plot( fontsize=12,
                  title='Indicator3 momentum')
img2.set_xlabel("Date")
img2.set_ylabel("Price")





# construct bbp helper frame
del bbp['SPY']
price_n = get_data([symbols], pd.date_range(start_date_train,end_date_test))

del price_n['SPY']
del top_band['SPY']
del bottom_band['SPY']
sma1, sma_n1, bbp1, top_band1, bottom_band1, momentum1 = caculate( symbols, sd=dt.datetime(2008,1,1), ed=dt.datetime(2011,12,31))
del sma1['SPY']
bbp_helper_frame = pd.concat([price_n,  top_band, bottom_band, sma1], axis=1)
bbp_helper_frame.columns=['Price','upper band','lower band','rolling mean']


img3=bbp_helper_frame.plot( fontsize=12,
                  title='Price, upper and lower bollinger bands')
img3.set_xlabel("Date")
img3.set_ylabel("Price")



# normalize bbp

bbp_n = bbp/bbp.ix[0]

bbp_n.columns = ['bbp JPM']

img4 = bbp_n.plot( fontsize=12,
                  title='Indicator2 BBP')
img4.set_xlabel("Date")
img4.set_ylabel("Normalized price")


plt.show()