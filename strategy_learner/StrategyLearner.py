"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import numpy as np
import QLearner as ql

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):


        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

        self.learner = ql.QLearner(num_states=3000, \
                                   num_actions=3, \
                                   alpha=0.2, \
                                   gamma=0.9, \
                                   rar=0.5, \
                                   radr=0.99, \
                                   dyna=0, \
                                   verbose=False)

        train_SMA = prices.rolling(window=14, min_periods=14).mean()
        train_SMA.fillna(method='ffill', inplace=True)
        train_SMA.fillna(method='bfill', inplace=True)
        train_P_SMA_ratio = prices / train_SMA
        train_rolling_std = prices.rolling(window=14, min_periods=14).std()
        top_band = train_SMA + (2 * train_rolling_std)
        bottom_band = train_SMA - (2 * train_rolling_std)
        train_bbp = (prices - bottom_band) / (top_band - bottom_band)
        # turn sma into price/sma ratio
        train_sma_ratio = prices / train_SMA

        # caculate momentum
        train_momentum = (prices / prices.copy().shift(14)) - 1

        train_daily_rets = (prices / prices.shift(1)) - 1
        train_vol = pd.rolling_std(train_daily_rets, 14)
        train_vol.fillna(method='ffill', inplace=True)
        train_vol.fillna(method='bfill', inplace=True)


       # dis

        train_P_SMA_ratio, train_bbp, train_momentum, train_vol = self.discritize(train_P_SMA_ratio, train_bbp, train_momentum, train_vol)

        train_states = train_bbp * 50 + train_P_SMA_ratio * 50 + train_momentum * 10 + train_vol
        start = train_states.index[0]
        end = train_states.index[-1]
        dates = pd.date_range(start, end)
        train_states = train_states.values
        Qframe = pd.DataFrame(index = dates)

        Qframe['Pos'] = 0
        Qframe['Price'] = prices.ix[start:end, symbol]
        Qframe['Cash'] = sv
        Qframe.fillna(method='ffill', inplace=True)
        Qframe.fillna(method='bfill', inplace=True)
        Qvalue = Qframe.values
        converged = False
        round = 0
        while not converged:

            p = 0
            state = p * 700 + train_states[0, 0]
            action = self.learner.querysetstate(state)
            total_days = train_states.shape[0]
            prev_val = sv
            for days in range(1, total_days):

                if p == 0 and action == 1:

                    Qvalue[days, 0] = -1000
                    Qvalue[days, 2] = Qvalue[days - 1, 2] + Qvalue[days, 1] * 1000
                    curr_val = Qvalue[days, 2] + Qvalue[days, 0] * Qvalue[days, 1]
                    p = 1
                elif p==0 and action == 2:

                    Qvalue[days, 0] = 1000
                    Qvalue[days, 2] = Qvalue[days - 1, 2] - Qvalue[days, 1] * 1000
                    curr_val = Qvalue[days, 2] + Qvalue[days, 0] * Qvalue[days, 1]
                    p = 2

                elif p == 1 and action == 2:

                    Qvalue[days, 0] = 1000
                    Qvalue[days, 2] = Qvalue[days - 1, 2] - Qvalue[days, 1] * 2000
                    curr_val = Qvalue[days, 2] + Qvalue[days, 0] * Qvalue[days, 1]
                    p = 2

                elif p == 2 and action == 1:

                    Qvalue[days, 0] = -1000
                    Qvalue[days, 2] = Qvalue[days - 1, 2] + Qvalue[days, 1] * 2000
                    curr_val = Qvalue[days, 2] + Qvalue[days, 0] * Qvalue[days, 1]
                    p = 1

                else:

                    Qvalue[days, 0] = Qvalue[days - 1, 0]
                    Qvalue[days, 2] = Qvalue[days - 1, 2]
                    curr_val = Qvalue[days, 2] + Qvalue[days, 0] * Qvalue[days, 1]

                reward = curr_val / prev_val - 1
                prev_val = curr_val
                state = p * 700 + train_states[days, 0]
                action = self.learner.query(state, reward)

            round += 1
            if round > 1000:
                converged = True


    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=100000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol]]
        trades = prices.copy()  # only portfolio symbols
        trades.values[:, :] = 0
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

        test_SMA = prices.rolling(window=14, min_periods=14).mean()
        test_SMA.fillna(method='ffill', inplace=True)
        test_SMA.fillna(method='bfill', inplace=True)
        test_P_SMA_ratio = prices / test_SMA
        test_rolling_std = prices.rolling(window=14, min_periods=14).std()
        top_band = test_SMA + (2 * test_rolling_std)
        bottom_band = test_SMA - (2 * test_rolling_std)
        test_bbp = (prices - bottom_band) / (top_band - bottom_band)
        # turn sma into price/sma ratio
        test_sma_ratio = prices / test_SMA

        # caculate momentum
        test_momentum = (prices / prices.copy().shift(14)) - 1

        test_daily_rets = (prices / prices.shift(1)) - 1
        test_vol = pd.rolling_std(test_daily_rets, 14)
        test_vol.fillna(method='ffill', inplace=True)
        test_vol.fillna(method='bfill', inplace=True)

        test_SMA_ratio_n, test_bbp_n, test_momentum_n, test_vol_n = self.discritize(test_P_SMA_ratio,test_bbp,test_momentum, test_vol)
        

        test_states = test_bbp_n * 50 + test_SMA_ratio_n * 50 + test_momentum_n * 10 + test_vol_n


        test_states = test_states.values
        position = 0

        for days in range(1, test_states.size):
            state = position * 1000 + test_states[days - 1, 0]
            action = self.learner.querysetstate(state)
            if position == 0 and action ==1:

                trades.values[days, :] = -1000
                position = 1
            elif position ==0 and action == 2:
                trades.values[days, :] = 1000
                position = 2


            elif position == 1 and action == 2:
                trades.values[days, :] = 2000
                position = 2

            elif position == 2 and action == 1:
                trades.values[days, :] = -2000
                position = 1


        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

    def discritize(self,SMA_ratio,bbp,momentum, vol ):

        SMA_ratio_n = SMA_ratio
        SMA_ratio_n.ix[:, 0] = np.digitize(SMA_ratio.ix[:, 0],
                                           np.linspace(SMA_ratio.ix[:, 0].min(), SMA_ratio.ix[:, 0].max(), 10)) - 1
        bbp_n = bbp
        bbp_n.ix[:, 0] = np.digitize(bbp.ix[:, 0], np.linspace(bbp.ix[:, 0].min(), bbp.ix[:, 0].max(), 10)) - 1

        momentum_n = momentum
        momentum_n.ix[:, 0] = np.digitize(momentum.ix[:, 0], np.linspace(momentum.ix[:, 0].min(), momentum.ix[:, 0].max(), 10)) - 1

        vol_n = vol
        vol_n.ix[:, 0] = np.digitize(vol.ix[:, 0], np.linspace(vol.ix[:, 0].min(), vol.ix[:, 0].max(), 10)) - 1
        return SMA_ratio_n,bbp_n,momentum_n, vol_n

    if __name__ == "__main__":
        print "One does not simply think up a strategy"
