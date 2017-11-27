"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import BagLearner as bag
import RTLearner as rt
import numpy as np
import random

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = bag.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 6}, bags=10, boost = False, verbose = False)

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # add your code to do learning here
        lookback =21

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume


        train_sma = prices.rolling(window=21, min_periods=21).mean()
        train_sma.fillna(method='ffill', inplace=True)
        train_sma.fillna(method='bfill', inplace=True)

        train_rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
        top_band = train_sma + (2 * train_rolling_std)
        bottom_band = train_sma - (2 * train_rolling_std)
        train_bbp = (prices - bottom_band) / (top_band - bottom_band)
        # turn sma into price/sma ratio
        train_sma_ratio = prices / train_sma

        # caculate momentum
        train_momentum = (prices / prices.copy().shift(lookback)) - 1


        df_ml = pd.concat([train_sma_ratio, train_bbp, train_momentum], axis=1)
        df_ml.columns = ['SMA_P', 'bbp', 'momentum']
        df_ml['n_return'] = np.nan

        ml_nday_rets = (prices / prices.shift(21)) - 1
        ml_nday_rets.ix[0:21,] = 0
        YBUY = 0.06
        YSELL = -0.06
        total_days = df_ml.shape[0]
        for i in range(0, total_days):
            if ml_nday_rets.ix[i,0] > YBUY:
                df_ml['n_return'].ix[i] = 1
            elif ml_nday_rets.ix[i,0] < YSELL:
                df_ml['n_return'].ix[i] = -1
            else:
                df_ml['n_return'].ix[i] = 0

        df_ml.fillna(method='ffill', inplace=True)
        df_ml.fillna(method='bfill', inplace=True)

        train_data = df_ml.as_matrix()
        self.learner.addEvidence(train_data[:total_days, 0:-1], train_data[:total_days, -1])



    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        lookback=21
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades.values[:, :] = 0
        prices=prices_all[[symbol,]]
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

        test_sma = prices.rolling(window=21, min_periods=21).mean()
        test_sma.fillna(method='ffill', inplace=True)
        test_sma.fillna(method='bfill', inplace=True)

        test_rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
        top_band = test_sma + (2 * test_rolling_std)
        bottom_band = test_sma - (2 * test_rolling_std)
        test_bbp = (prices - bottom_band) / (top_band - bottom_band)
        # turn sma into price/sma ratio
        test_sma_ratio = prices / test_sma

        # caculate momentum
        test_momentum = (prices / prices.copy().shift(lookback)) - 1



        df_ml_t = pd.concat([test_sma_ratio, test_bbp, test_momentum], axis=1)
        df_ml_t.columns = ['SMA_P', 'bbp', 'momentum']
        df_ml_t['n_return'] = np.nan
        test_days = df_ml_t.shape[0]
        test_data = df_ml_t.as_matrix()

        testX=test_data[:test_days, 0:-1]
        test_res = self.learner.query(testX)
        test_res = test_res.astype(int)
        total_days = testX.shape[0]
        p=0
        status = 0
        for i in range(1, len(test_res)):

            if p == 0 and test_res[i] == -1:

                status= -1000
                p = -1
            elif p == 0 and test_res[i] == 1:
                status = 1000
                p = 1

            elif p == 1 and test_res[i] == -1:
                status = -2000
                p = -1

            elif p == -1 and test_res[i] == 1:
                status = 2000
                p = 1
            else:
                status = 0
            trades.values[i,:] = status


        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades






if __name__=="__main__":
    print "One does not simply think up a strategy"
