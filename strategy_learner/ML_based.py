import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
import marketsimcode as mk
import BagLearner as bag
import RTLearner as rt
import rule_based as rule

start_date_train = '2008-01-01'
end_date_train = '2009-12-31'
start_date_test = '2010-01-01'
end_date_test = '2011-12-31'

syms =['JPM']

sma, bbp, momentum,orders = rule.caculate(symbols=syms, sdate='2008-01-01', edate='2009-12-31',  lookback=21)
manual_ml = sma
manual_ml['sma'] = np.nan
manual_ml['sma'] = sma
manual_ml['bbp'] = np.nan
manual_ml['momentum'] = np.nan
manual_ml['bbp'] = bbp
manual_ml['momentum'] = momentum

price = get_data(syms, pd.date_range(start_date_train, end_date_train))

manual_ml['return'] = np.nan
manual_ml['return'].ix[0:-21] = price['JPM'].ix[21:manual_ml.shape[0]].values / price['JPM'].ix[0:-21].values - 1
manual_ml['return'].ix[-21:] = 0

YBUY = 0.05
YSELL = -0.05
for i in range(manual_ml.shape[0]):
    if manual_ml['return'].ix[i] < YSELL:
        manual_ml['return'].ix[i] = -1
    elif manual_ml['return'].ix[i] > YBUY:
        manual_ml['return'].ix[i] = 1
    else:
        manual_ml['return'].ix[i] = 0

del manual_ml['SPY']
del manual_ml['JPM']

manual_ml.fillna(method='ffill', inplace=True)
manual_ml.fillna(method='bfill', inplace=True)

array_ml = manual_ml.as_matrix()

trainX = array_ml[:, 0:-1]
trainY = array_ml[:, -1]
trainX= np.array(trainX)
trainY= np.array(trainY)
print trainX, trainY
learner = bag.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5, "verbose" : False}, bags=15, boost = False, verbose = False)
learner.addEvidence(trainX, trainY)
predY = learner.query(trainX)
manual_ml['predY'] = predY

print manual_ml

