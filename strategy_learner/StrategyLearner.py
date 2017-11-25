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


        bins_bbp = np.linspace(train_bbp.ix[:, 0].min(), train_bbp.ix[:, 0].max(), 10)
        train_bbp.ix[:, 0] = np.digitize(train_bbp.ix[:, 0], bins_bbp) - 1

        bins_momentum = np.linspace(train_momentum.ix[:, 0].min(), train_momentum.ix[:, 0].max(), 10)
        train_momentum.ix[:, 0] = np.digitize(train_momentum.ix[:, 0], bins_momentum) - 1

        bins_P_SMA_ratio = np.linspace(train_P_SMA_ratio.ix[:, 0].min(), train_P_SMA_ratio.ix[:, 0].max(), 10)
        train_P_SMA_ratio.ix[:, 0] = np.digitize(train_P_SMA_ratio.ix[:, 0], bins_P_SMA_ratio) - 1

        bins_vol = np.linspace(train_vol.ix[:, 0].min(), train_vol.ix[:, 0].max(), 10)
        train_vol.ix[:, 0] = np.digitize(train_vol.ix[:, 0], bins_vol) - 1

        train_states = train_bbp * 50 + train_P_SMA_ratio * 50 + train_momentum * 10 + train_vol
        start = train_states.index[0]
        end = train_states.index[-1]
        dates = pd.date_range(start, end)
        train_states = train_states.values
        Qframe = pd.DataFrame(index = dates)

        Qframe['Pos'] = 0
        Qframe['Price'] = prices.ix[start:end, symbol]
        Qframe['Cash'] = sv
        Qframe['P_V'] = sv
        Qframe.fillna(method='ffill', inplace=True)
        Qframe.fillna(method='bfill', inplace=True)

        converged = False
        round = 0
        while not converged:

            position = 0
            state = position * 700 + train_states[0, 0]
            action = self.learner.querysetstate(state)
            total_days = train_states.shape[0]
            for days in range(1, total_days):
                if position == 0:
                    if action == 1:
                        position = 1
                        Qframe.ix[days, 0] = -1000
                        Qframe.ix[days, 2] = Qframe.ix[days - 1, 2] + Qframe.ix[days, 1] * 1000
                    elif action == 2:
                        position = 2
                        Qframe.ix[days, 0] = 1000
                        Qframe.ix[days, 2] = Qframe.ix[days - 1, 2] - Qframe.ix[days, 1] * 1000
                    else:
                        position = 0
                        Qframe.ix[days, 0] = Qframe.ix[days - 1, 0]
                        Qframe.ix[days, 2] = Qframe.ix[days - 1, 2]

                elif position == 1 and action == 2:
                    position = 2
                    Qframe.ix[days, 0] = 1000
                    Qframe.ix[days, 2] = Qframe.ix[days - 1, 2] - Qframe.ix[days, 1] * 2000

                elif position == 2 and action == 1:
                    position = 1
                    Qframe.ix[days, 0] = -1000
                    Qframe.ix[days, 2] = Qframe.ix[days - 1, 2] + Qframe.ix[days, 1] * 2000

                else:
                    position = position
                    Qframe.ix[days, 0] = Qframe.ix[days - 1, 0]
                    Qframe.ix[days, 2] = Qframe.ix[days - 1, 2]

                Qframe.ix[days, 3] = Qframe.ix[days, 2] + Qframe.ix[days, 0] * Qframe.ix[days, 1]
                reward = Qframe.ix[days, 3] / Qframe.ix[days - 1, 3] - 1
                state = position * 1000 + train_states[days, 0]
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

        test_SMA = pd.rolling_mean(prices, 15).fillna(method='bfill')
        test_stdev = pd.rolling_std(prices, 15).fillna(method='bfill')
        test_P_SMA_ratio = prices / test_SMA
        test_momentum = (prices / prices.shift(15) - 1).fillna(method='bfill')
        test_bbp = (prices - test_SMA + 2 * test_stdev) / (4 * test_stdev)
        test_daily_rets = (prices / prices.shift(1)) - 1
        test_vol = pd.rolling_std(test_daily_rets, 15).fillna(method='bfill')

        '''DISCRETIZE'''
        bins_bbp = np.linspace(test_bbp.ix[:, 0].min(), test_bbp.ix[:, 0].max(), 10)
        test_bbp.ix[:, 0] = np.digitize(test_bbp.ix[:, 0], bins_bbp) - 1

        bins_momentum = np.linspace(test_momentum.ix[:, 0].min(), test_momentum.ix[:, 0].max(), 10)
        test_momentum.ix[:, 0] = np.digitize(test_momentum.ix[:, 0], bins_momentum) - 1

        bins_P_SMA_ratio = np.linspace(test_P_SMA_ratio.ix[:, 0].min(), test_P_SMA_ratio.ix[:, 0].max(), 10)
        test_P_SMA_ratio.ix[:, 0] = np.digitize(test_P_SMA_ratio.ix[:, 0], bins_P_SMA_ratio) - 1

        bins_vol = np.linspace(test_vol.ix[:, 0].min(), test_vol.ix[:, 0].max(), 10)
        test_vol.ix[:, 0] = np.digitize(test_vol.ix[:, 0], bins_vol) - 1

        test_indicator = pd.concat([test_bbp, test_momentum, test_vol],
                                   keys=['Ind1', 'Ind2', 'Ind3'], axis=1)

        test_states = test_indicator['Ind1'] * 100 + \
                      test_indicator['Ind2'] * 10 + \
                      test_indicator['Ind3']

        '''TEST'''
        test_states = test_states.values
        position = 0

        for days in range(1, test_states.size):
            state = position * 1000 + test_states[days - 1, 0]
            action = self.learner.querysetstate(state)
            if position == 0:
                if action == 1:
                    trades.values[days, :] = -1000
                    position = 1
                elif action == 2:
                    trades.values[days, :] = 1000
                    position = 2
                else:
                    position = 0

            elif position == 1 and action == 2:
                trades.values[days, :] = 2000
                position = 2

            elif position == 2 and action == 1:
                trades.values[days, :] = -2000
                position = 1

            else:
                position = position

        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

    if __name__ == "__main__":
        print "One does not simply think up a strategy"
