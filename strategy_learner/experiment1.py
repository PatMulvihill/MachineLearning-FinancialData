'''Author: Lu Wang lwang496'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import ManualStrategy as mn
import os
import strategyexperiment1 as sl

import datetime as dt
from util import get_data, plot_data
import marketsimcode as mk


start_date_train = dt.datetime(2008,1,01)
end_date_train = dt.datetime(2009,12,31)
start_date_test = dt.datetime(2010,1,01)
end_date_test = dt.datetime(2011,12,31)

syms ='JPM'

strategy_learner = sl.StrategyLearner( verbose = False, impact=0.0)
strategy_learner.addEvidence(symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)
strategy_trade = strategy_learner.testPolicy(symbol = "JPM", \
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

order_list_manual = mn.testPolicy(symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)
print order_list_manual


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
manual_portvals = mk.compute_portvals(order_list_manual, start_val=100000, commission=0, impact=0)
manual_portvals.fillna(method='ffill', inplace=True)
manual_portvals.fillna(method='bfill', inplace=True)

strategy_portvals = mk.compute_portvals(strategy_order_n, start_val=100000, commission=0, impact=0)
strategy_portvals.fillna(method='ffill', inplace=True)
strategy_portvals.fillna(method='bfill', inplace=True)


final_frame = pd.concat([manual_portvals, strategy_portvals, benchmark_frame], axis=1)
final_frame.columns = ['Manual Strategy', 'StrategyLearner','Benchmark']

print final_frame

final_frame.fillna(method='ffill', inplace=True)
final_frame.fillna(method='bfill', inplace=True)
final_frame= final_frame/final_frame.ix[0]
img= final_frame.plot( fontsize=12,
                  title='Manual Strategy vs Benchmark (in sample)')
img.set_xlabel("Date")
img.set_ylabel("Normalized value")

bench_daily_rets = (benchmark_frame / benchmark_frame.shift(1)) - 1
bench_daily_rets = bench_daily_rets[1:]
bench_cr = (benchmark_frame.ix[-1] / benchmark_frame.ix[0]) - 1
bench_adr = bench_daily_rets.mean()
bench_sddr = bench_daily_rets.std()

print "Benchmark Volatility (stdev of daily returns):", bench_sddr
print "Benchmark Average Daily Return:", bench_adr
print "Benchmark Cumulative Return:", bench_cr


daily_rets = (manual_portvals / manual_portvals.shift(1)) - 1
daily_rets = daily_rets[1:]
cr = (manual_portvals.ix[-1] / manual_portvals.ix[0]) - 1
adr = daily_rets.mean()
sddr = daily_rets.std()
sr = np.sqrt(252) * (adr) / sddr

print "manual strategy in sample Volatility (stdev of daily returns):", sddr
print "manual strategy in sample Average Daily Return:", adr
print "manual strategy in sample Cumulative Return:", cr

s_daily_rets = (strategy_portvals / strategy_portvals.shift(1)) - 1
s_daily_rets = s_daily_rets[1:]
s_cr = (strategy_portvals.ix[-1] / strategy_portvals.ix[0]) - 1
s_adr = s_daily_rets.mean()
s_sddr = s_daily_rets.std()
s_sr = np.sqrt(252) * (s_adr) / s_sddr

print " strategy learner in sample Volatility (stdev of daily returns):", s_sddr
print " strategy learner in sample Average Daily Return:", s_adr
print " strategy learner in sample Cumulative Return:", s_cr


plt.show()