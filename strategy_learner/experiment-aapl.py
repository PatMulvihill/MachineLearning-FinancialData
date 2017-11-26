

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import ManualStrategy as mn
import os
import StrategyLearner as sl

import datetime as dt
from util import get_data, plot_data
import marketsimcode as mk



start_date_train = dt.datetime(2008,1,01)
end_date_train = dt.datetime(2009,12,31)
start_date_test = dt.datetime(2010,1,01)
end_date_test = dt.datetime(2011,12,31)

syms ='AAPL'

strategy_learner = sl.StrategyLearner( verbose = False, impact=0.0)
strategy_learner.addEvidence(symbol = "AAPL", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)
strategy_trade = strategy_learner.testPolicy(symbol = "AAPL", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)

strategy_order_list = []
for day in strategy_trade.index:

    if strategy_trade.ix[day, 0] == 1000:
        strategy_order_list.append([day.date(), syms, 'BUY', 1000])
    elif strategy_trade.ix[day, 0] == 2000:
        strategy_order_list.append([day.date(), syms, 'BUY', 2000])
    elif strategy_trade.ix[day, 0] == -2000:
        strategy_order_list.append([day.date(), syms, 'SELL', 2000])
    elif strategy_trade.ix[day, 0] == -1000:
        strategy_order_list.append([day.date(), syms, 'SELL', 1000])

strategy_order_list.append([end_date_train.date(), syms, 'SELL', 0])

strategy_order_n = pd.DataFrame(np.array(strategy_order_list), columns=['Date', 'Symbol', 'Order', 'Shares'])
strategy_order_n.set_index('Date', inplace=True)

print strategy_trade
print strategy_order_n




price = get_data([syms], pd.date_range(start_date_train, end_date_train))
benchmark_frame = price
benchmark_frame['Benchmark'] = np.nan
benchmark_frame['Benchmark'].ix[0] = 100000

for i in range(0, benchmark_frame.shape[0]):
    benchmark_frame['Benchmark'].ix[i] = 100000 - 1000 * price['AAPL'].ix[0] + \
                                          1000 * price['AAPL'].ix[i]

del benchmark_frame['SPY']
del benchmark_frame['AAPL']
bench_daily_rets = (benchmark_frame / benchmark_frame.shift(1)) - 1
bench_daily_rets = bench_daily_rets[1:]
bench_cr = (benchmark_frame.ix[-1] / benchmark_frame.ix[0]) - 1
bench_adr = bench_daily_rets.mean()
bench_sddr = bench_daily_rets.std()

print "Benchmark Volatility (stdev of daily returns):", bench_sddr
print "Benchmark Average Daily Return:", bench_adr
print "Benchmark Cumulative Return:", bench_cr


strategy_portvals = mk.compute_portvals(strategy_order_n, start_val=100000, commission=0, impact=0)
strategy_portvals.fillna(method='ffill', inplace=True)
strategy_portvals.fillna(method='bfill', inplace=True)


s_daily_rets = (strategy_portvals / strategy_portvals.shift(1)) - 1
s_daily_rets = s_daily_rets[1:]
s_bench_cr = (strategy_portvals.ix[-1] / strategy_portvals.ix[0]) - 1
s_bench_adr = s_daily_rets.mean()
s_bench_sddr = s_daily_rets.std()

print "s Volatility (stdev of daily returns):", s_bench_sddr
print "s Average Daily Return:", s_bench_adr
print "s Cumulative Return:", s_bench_cr

final_frame = pd.concat([ strategy_portvals, benchmark_frame], axis=1)
final_frame.columns = [ 'StrategyLearner','Benchmark']

print final_frame

final_frame.fillna(method='ffill', inplace=True)
final_frame.fillna(method='bfill', inplace=True)
final_frame= final_frame/final_frame.ix[0]
img= final_frame.plot( fontsize=12,
                  title='Manual Strategy vs Benchmark (in sample)')
img.set_xlabel("Date")
img.set_ylabel("Normalized value")
plt.show()