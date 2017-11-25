import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
import marketsimcode as mk
import datetime as dt

start_date_train = dt.datetime(2008,1,01)
end_date_train = dt.datetime(2009,12,31)
start_date_test = dt.datetime(2010,1,01)
end_date_test = dt.datetime(2011,12,31)

def testPolicy(symbol , sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
    date_range = pd.date_range(sd, ed)
    date_range = pd.date_range(sd, ed)
    sym = [symbol]
    price = get_data(sym, pd.date_range(sd, ed))



    frame = price.copy()

    frame['Benchmark'] = np.nan
    frame['Benchmark'].ix[0] = 100000

    frame['Position'] = np.nan
    frame['BUY'] = 0
    frame['SELL'] = 0
    frame['BUY or SELL'] = np.nan

    for i in range(0, frame.shape[0]):
        frame['Benchmark'].ix[i] = 100000 - 1000 * frame[symbol].ix[0] + 1000 * frame[symbol].ix[i]

    if frame[symbol].ix[1] > frame[symbol].ix[0]:
        frame['Position'].ix[0] = 1000
        frame['BUY'].ix[0] = 1000

    elif frame[symbol].ix[1] < frame[symbol].ix[0]:
        frame['Position'].ix[0] = -1000
        frame['SELL'].ix[0] = 1000
    else:
        frame[symbol].ix[1] = frame[symbol].ix[0]



    for i in range(0, frame.shape[0] - 1):
        if frame[symbol].ix[i + 1] > frame[symbol].ix[i]:
            if frame['Position'].ix[i - 1] == 1000:
                frame['Position'].ix[i] = 1000
            elif frame['Position'].ix[i - 1] == -1000:
                frame['Position'].ix[i] = 1000
                frame['BUY'].ix[i] = 2000
            elif frame['Position'].ix[i - 1] == 0:
                frame['Position'].ix[i] = 1000
                frame['BUY'].ix[i] = 1000

        elif frame[symbol].ix[i + 1] < frame[symbol].ix[i]:
            if frame['Position'].ix[i - 1] == 1000:
                frame['Position'].ix[i] = -1000
                frame['SELL'].ix[i] = 2000

            elif frame['Position'].ix[i - 1] == -1000:
                frame['Position'].ix[i] = -1000

            elif frame['Position'].ix[i - 1] == 0:
                frame['Position'].ix[i] = -1000
                frame['SELL'].ix[i] = 1000


        else:
            if i==0:
                frame['Position'].ix[i] = 0
            else:
                frame['Position'].ix[i] = frame['Position'].ix[i-1]

    order_list = []
    for day in frame.index:

        if frame.ix[day, 'BUY'] == 1000:
            order_list.append([day.date(), symbol, 'BUY', 1000])
        elif frame.ix[day, 'BUY'] == 2000:
            order_list.append([day.date(), symbol, 'BUY', 2000])
        elif frame.ix[day, 'SELL'] == 2000:
            order_list.append([day.date(), symbol, 'SELL', 2000])
        elif frame.ix[day, 'SELL'] == 1000:
            order_list.append([day.date(), symbol, 'SELL', 1000])

    order_list_final = pd.DataFrame(np.array(order_list), columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list_final.set_index('Date', inplace=True)

    return  order_list_final

def author():
    return 'lwang496'

sym = 'JPM'
order_list_final  = testPolicy(symbol=sym, sd=dt.datetime(2008, 1, 02), ed=dt.datetime(2009,12,31), sv=100000)

# generate benchmark frame
price = get_data([sym], pd.date_range(start_date_train, end_date_train))
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
print benchmark_frame
benchmark_frame = benchmark_frame/benchmark_frame.ix[0]

#generate beststrategy frame
best_strategy_frame = mk.compute_portvals(order_list_final, start_val=100000, commission=0, impact=0)
daily_rets = (best_strategy_frame / best_strategy_frame.shift(1)) - 1
daily_rets = daily_rets[1:]
cr = (best_strategy_frame.ix[-1] / best_strategy_frame.ix[0]) - 1
adr = daily_rets.mean()
sddr = daily_rets.std()

print "Volatility (stdev of daily returns):", sddr
print "Average Daily Return:", adr
print "Cumulative Return:", cr


best_strategy_frame = best_strategy_frame/best_strategy_frame.ix[0]

final_frame = pd.concat([best_strategy_frame, benchmark_frame], axis=1)
final_frame.columns = ['Best Strategy', 'Benchmark']

final_frame.fillna(method='ffill', inplace=True)
final_frame.fillna(method='bfill', inplace=True)
img = final_frame.plot( fontsize=12,color=['k', 'b'],
                  title='Best Strategy vs Benchmark')
img.set_xlabel("Date")
img.set_ylabel("Normalized value")

plt.show()

print final_frame


