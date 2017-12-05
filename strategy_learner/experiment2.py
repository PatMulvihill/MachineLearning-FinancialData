"""Author: Lu Wang, lwang496, lwang496@gatech.edu"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import ManualStrategy as mn
import os
import datetime as dt
from util import get_data, plot_data
import marketsimcode as mk



"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""


# Author: Lu Wang lwang496

import datetime as dt
import pandas as pd
import util as ut
import random
import numpy as np
import QLearner as ql
import strategyexperiment1 as sl


seed = 141109000
np.random.seed(seed)
random.seed(seed)

start_date_train = dt.datetime(2008,1,01)
end_date_train = dt.datetime(2009,12,31)
start_date_test = dt.datetime(2010,1,01)
end_date_test = dt.datetime(2011,12,31)

syms ='JPM'

strategy_learner = sl.StrategyLearner1( verbose = False, impact=0.0)
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

strategy_portvals1 = mk.compute_portvals(strategy_order_n, start_val=100000, commission=0, impact=0)
strategy_portvals1.fillna(method='ffill', inplace=True)
strategy_portvals1.fillna(method='bfill', inplace=True)
print "strategy portvals"
print strategy_order_n


seed = 141109000
np.random.seed(seed)
random.seed(seed)

strategy_learner1 = sl.StrategyLearner1( verbose = False, impact=0.005)
strategy_learner1.addEvidence(symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)
strategy_trade1 = strategy_learner1.testPolicy(symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)

strategy_order_list = []
for day in strategy_trade1.index:

    if strategy_trade1.ix[day, 0] == 1000:
        strategy_order_list.append([day.date(), syms, 'BUY', 1000])
    elif strategy_trade1.ix[day, 0] == 2000:
        strategy_order_list.append([day.date(), syms, 'BUY', 2000])
    elif strategy_trade1.ix[day, 0] == -2000:
        strategy_order_list.append([day.date(), syms, 'SELL', 2000])
    elif strategy_trade1.ix[day, 0] == -1000:
        strategy_order_list.append([day.date(), syms, 'SELL', 1000])

strategy_order_list.append([end_date_train.date(), syms, 'SELL', 0])

strategy_order_n2 = pd.DataFrame(np.array(strategy_order_list), columns=['Date', 'Symbol', 'Order', 'Shares'])
strategy_order_n2.set_index('Date', inplace=True)
print strategy_order_n2



seed = 141109000
np.random.seed(seed)
random.seed(seed)

strategy_learner1 = sl.StrategyLearner1( verbose = False, impact=0.05)
strategy_learner1.addEvidence(symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)
strategy_trade1 = strategy_learner1.testPolicy(symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000)

strategy_order_list = []
for day in strategy_trade1.index:

    if strategy_trade1.ix[day, 0] == 1000:
        strategy_order_list.append([day.date(), syms, 'BUY', 1000])
    elif strategy_trade1.ix[day, 0] == 2000:
        strategy_order_list.append([day.date(), syms, 'BUY', 2000])
    elif strategy_trade1.ix[day, 0] == -2000:
        strategy_order_list.append([day.date(), syms, 'SELL', 2000])
    elif strategy_trade1.ix[day, 0] == -1000:
        strategy_order_list.append([day.date(), syms, 'SELL', 1000])

strategy_order_list.append([end_date_train.date(), syms, 'SELL', 0])

strategy_order_n3 = pd.DataFrame(np.array(strategy_order_list), columns=['Date', 'Symbol', 'Order', 'Shares'])
strategy_order_n3.set_index('Date', inplace=True)


price = get_data([syms], pd.date_range(start_date_train, end_date_train))




strategy_portvals2 = mk.compute_portvals(strategy_order_n2, start_val=100000, commission=0, impact=0.005)
strategy_portvals2.fillna(method='ffill', inplace=True)
strategy_portvals2.fillna(method='bfill', inplace=True)

strategy_portvals3 = mk.compute_portvals(strategy_order_n3, start_val=100000, commission=0, impact=0.05)
strategy_portvals3.fillna(method='ffill', inplace=True)
strategy_portvals3.fillna(method='bfill', inplace=True)

final_frame = pd.concat([strategy_portvals1, strategy_portvals2, strategy_portvals3], axis=1)
final_frame.columns = ['impact=0', 'impact=0.005','impact=0.05']



final_frame.fillna(method='ffill', inplace=True)
final_frame.fillna(method='bfill', inplace=True)
final_frame= final_frame/final_frame.ix[0]
print final_frame
img= final_frame.plot( fontsize=12,
                  title='Impact affect the trade')
img.set_xlabel("Date")
img.set_ylabel("Normalized value")


s_daily_rets1 = (strategy_portvals1 / strategy_portvals1.shift(1)) - 1
s_daily_rets1 = s_daily_rets1[1:]
s_cr1 = (strategy_portvals1.ix[-1] / strategy_portvals1.ix[0]) - 1
s_adr1 = s_daily_rets1.mean()
s_sddr1 = s_daily_rets1.std()
s_sr1 = np.sqrt(252) * (s_adr1) / s_sddr1

print " strategy learner in sample Volatility (stdev of daily returns):", s_sddr1
print " strategy learner in sample Average Daily Return:", s_adr1
print " strategy learner in sample Cumulative Return:", s_cr1

s_daily_rets2 = (strategy_portvals2 / strategy_portvals2.shift(1)) - 1
s_daily_rets2 = s_daily_rets2[1:]
s_cr2 = (strategy_portvals2.ix[-1] / strategy_portvals2.ix[0]) - 1
s_adr2 = s_daily_rets2.mean()
s_sddr2 = s_daily_rets2.std()
s_sr2 = np.sqrt(252) * (s_adr2) / s_sddr2

print " strategy learner in sample Volatility (stdev of daily returns):", s_sddr2
print " strategy learner in sample Average Daily Return:", s_adr2
print " strategy learner in sample Cumulative Return:", s_cr2


s_daily_rets3 = (strategy_portvals3 / strategy_portvals3.shift(1)) - 1
s_daily_rets3 = s_daily_rets3[1:]
s_cr3 = (strategy_portvals3.ix[-1] / strategy_portvals3.ix[0]) - 1
s_adr3 = s_daily_rets3.mean()
s_sddr3 = s_daily_rets3.std()
s_sr3 = np.sqrt(252) * (s_adr3) / s_sddr3

print " strategy learner in sample Volatility (stdev of daily returns):", s_sddr3
print " strategy learner in sample Average Daily Return:", s_adr3
print " strategy learner in sample Cumulative Return:", s_cr3


plt.show()

