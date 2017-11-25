import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
import marketsim as mk

start_date_train = '2008-01-01'
end_date_train = '2009-12-31'
start_date_test = '2010-01-01'
end_date_test = '2011-12-31'



def caculate( symbols, sdate='2008-01-01', edate='2009-12-31',  lookback=21):

    price = get_data(symbols, pd.date_range(sdate,edate))

    sma = price.rolling(window = lookback, min_periods = lookback).mean()


    # caculate bbp
    rolling_std = price.rolling(window = lookback, min_periods = lookback).std()
    top_band = sma + (2*rolling_std)
    bottom_band = sma - (2*rolling_std)
    bbp = (price - bottom_band) / (top_band- bottom_band)
    # turn sma into price/sma ratio
    sma = price/sma

    #caculate momentum
    momentum = (price / price.copy().shift(lookback))-1

    # orders
    orders = price.copy()
    orders.ix[:,:] = np.NaN
    # create a binary array showing when price is above sma
    sma_cross = pd.DataFrame(0,index = sma.index, columns = sma.columns)
    sma_cross[sma >= 1] =1
    #turn that array into that only shows the crossings
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0
    # caculate the result of the overall strategy
    orders[(sma < 0.95) & (bbp < 0)] = 100
    orders[(sma > 1.05) & (bbp > 1)] = -100
    orders[(sma_cross != 0)] = 0
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)


    orders[1:] = orders.diff()
    orders.ix[0] = 0

    orders = orders.loc[(orders != 0).any(axis=1)]

    order_list = []
    for day in orders.index:
        print day
        for sym in symbols:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, 'BUY', 100])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, 'SELL', 100])

    return sma, bbp, momentum, order_list


syms =['JPM']

sma, bbp, momentum,orders = caculate(symbols=syms, sdate='2008-01-01', edate='2009-12-31',  lookback=21)
for order in orders:
    print "      ".join(str(x) for x in order)

# convert order list to padnas frame
df_manual_strategy = pd.DataFrame(np.array(orders), columns=[ 'Date','Symbol', 'Order', 'Shares'])
df_manual_strategy.to_csv("order_manual_strategy.csv")
start_date = df_manual_strategy.ix[0]['Date']
end_date = df_manual_strategy.iloc[-1]['Date']
print end_date
manual = pd.read_csv("order_manual_strategy.csv", index_col='Date',parse_dates=True, na_values=['nan'])
del manual['Unnamed: 0']
manual.to_csv("order_manual.csv")
print manual


portvals = mk.compute_portvals(orders_file='order_manual.csv', start_val=100000, commission=9.95, impact=0.005)

print portvals

sma_plot = sma.plot( fontsize=12, title='SMA ratio')
bbp_plot = bbp.plot( fontsize=12, title='Bollinger Bands (21 days) %')
momentum_plot = momentum.plot( fontsize=12, title='Normalized momentum ')


# compare with benchmark
price = get_data(syms, pd.date_range(start_date_train, end_date_train))
benchmark_frame = price
benchmark_frame['Benchmark'] = np.nan
benchmark_frame['Benchmark'].ix[0] = 100000

for i in range(0, benchmark_frame.shape[0]):
    benchmark_frame['Benchmark'].ix[i] = 100000 - 1000 * price['JPM'].ix[0] + \
                                          1000 * price['JPM'].ix[i]

del benchmark_frame['SPY']
del benchmark_frame['JPM']


bench_daily_rets = (benchmark_frame / benchmark_frame.shift(1)) - 1
bench_daily_rets = bench_daily_rets[1:]
bench_cr = (benchmark_frame.ix[-1] / benchmark_frame.ix[0]) - 1
bench_adr = bench_daily_rets.mean()
bench_sddr = bench_daily_rets.std()

print "Benchmark Volatility (stdev of daily returns):", bench_sddr
print "Benchmark Average Daily Return:", bench_adr
print "Benchmark Cumulative Return:", bench_cr


# get daily return of the portvals
daily_rets = (portvals / portvals.shift(1)) - 1
daily_rets = daily_rets[1:]
cr = (portvals.ix[-1] / portvals.ix[0]) - 1
adr = daily_rets.mean()
sddr = daily_rets.std()

print "Volatility (stdev of daily returns):", sddr
print "Average Daily Return:", adr
print "Cumulative Return:", cr

# normalize
portvals_n=portvals/portvals.ix[0]
benchmark_frame_n = benchmark_frame/benchmark_frame.ix[0]

new_frame = benchmark_frame_n.merge(portvals_n, how='outer', left_index=True, right_index=True)
new_frame.fillna(method = 'ffill', inplace = True)
new_frame.fillna(method='bfill', inplace=True)
print new_frame


plt.show()














