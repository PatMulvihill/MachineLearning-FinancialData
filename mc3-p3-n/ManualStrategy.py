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

    

    order_list_final = pd.DataFrame(np.array(order_list), columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list_final.set_index('Date', inplace=True)
    return order_list_final


syms ='JPM'

order_list = testPolicy(symbol=syms, sd=start_date_train, ed=end_date_train,sv = 100000)

print order_list
# compare with benchmark
price = get_data([syms], pd.date_range(start_date_train, end_date_train))
benchmark_frame = price
benchmark_frame['Benchmark'] = np.nan
benchmark_frame['Benchmark'].ix[0] = 100000

for i in range(0, benchmark_frame.shape[0]):
    benchmark_frame['Benchmark'].ix[i] = 100000 - 1000 * price['JPM'].ix[0] + \
                                          1000 * price['JPM'].ix[i]

del benchmark_frame['SPY']
del benchmark_frame['JPM']

#use marketsimcode to caculate the money
portvals = mk.compute_portvals(order_list, start_val=100000, commission=9.95, impact=0.005)
portvals.fillna(method='ffill', inplace=True)
portvals.fillna(method='bfill', inplace=True)


final_frame = pd.concat([portvals, benchmark_frame], axis=1)
final_frame.columns = ['Manual Strategy', 'Benchmark']


final_frame.fillna(method='ffill', inplace=True)
final_frame.fillna(method='bfill', inplace=True)
final_frame= final_frame/final_frame.ix[0]
img= final_frame.plot( fontsize=12,color=['k', 'b'],
                  title='Manual Strategy vs Benchmark (in sample)')
img.set_xlabel("Date")
img.set_ylabel("Normalized value")
print final_frame

for index, row in order_list.iterrows():
    if row['Order'] == 'SELL':

        plt.axvline(x=index, color='red', linestyle='-')
    if row['Order'] == 'BUY':

        plt.axvline(x=index, color='green', linestyle='-')

plt.show()

# out of sample


order_list_n = testPolicy(symbol=syms, sd=start_date_test, ed=end_date_test,sv = 100000)
price_n = get_data([syms], pd.date_range(start_date_test, end_date_test))
benchmark_frame_n = price_n
benchmark_frame_n['Benchmark'] = np.nan
benchmark_frame_n['Benchmark'].ix[0] = 100000
for i in range(0, benchmark_frame_n.shape[0]):
    benchmark_frame_n['Benchmark'].ix[i] = 100000 - 1000 * price_n['JPM'].ix[0] + \
                                          1000 * price_n['JPM'].ix[i]

del benchmark_frame_n['SPY']
del benchmark_frame_n['JPM']

portvals_n = mk.compute_portvals(order_list_n, start_val=100000, commission=9.95, impact=0.005)



final_frame_n = pd.concat([portvals_n, benchmark_frame_n], axis=1)
final_frame_n.columns = ['Manual Strategy', 'Benchmark']


final_frame_n.fillna(method='ffill', inplace=True)
final_frame_n.fillna(method='bfill', inplace=True)
final_frame_n= final_frame_n/final_frame_n.ix[0]
img_n= final_frame_n.plot( fontsize=12,color=['k', 'b'],
                  title='Manual Strategy vs Benchmark (out of sample)')
img_n.set_xlabel("Date")
img_n.set_ylabel("Normalized value")

for index, row in order_list_n.iterrows():
    if row['Order'] == 'SELL':

        plt.axvline(x=index, color='red', linestyle='-')
    if row['Order'] == 'BUY':

        plt.axvline(x=index, color='green', linestyle='-')





daily_rets = (portvals / portvals.shift(1)) - 1
daily_rets = daily_rets[1:]
cr = (portvals.ix[-1] / portvals.ix[0]) - 1
adr = daily_rets.mean()
sddr = daily_rets.std()
sr = np.sqrt(252) * (adr) / sddr
print portvals
print "manual strategy in sample Volatility (stdev of daily returns):", sddr
print "manual strategy in sample Average Daily Return:", adr
print "manual strategy in sample Cumulative Return:", cr
print "sharp ratio insample", sr


daily_rets_n = (portvals_n / portvals_n.shift(1)) - 1
daily_rets_n = daily_rets_n[1:]
cr_n = (portvals_n.ix[-1] / portvals_n.ix[0]) - 1
adr_n = daily_rets_n.mean()
sddr_n = daily_rets_n.std()
sr_n = np.sqrt(252) * (adr_n) / sddr_n
print portvals_n
print "manual strategy out sample Volatility (stdev of daily returns):", sddr_n
print "manual strategy out sample Average Daily Return:", adr_n
print "manual strategy out sample Cumulative Return:", cr_n
print "sharp ratio out sample", sr_n


bench_daily_rets = (benchmark_frame / benchmark_frame.shift(1)) - 1
bench_daily_rets = bench_daily_rets[1:]
bench_cr = (benchmark_frame.ix[-1] / benchmark_frame.ix[0]) - 1
bench_adr = bench_daily_rets.mean()
bench_sddr = bench_daily_rets.std()

print "Benchmark Volatility (stdev of daily returns):", bench_sddr
print "Benchmark Average Daily Return:", bench_adr
print "Benchmark Cumulative Return:", bench_cr


plt.show()






