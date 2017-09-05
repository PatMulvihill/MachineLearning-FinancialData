"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import sys
sys.path.append('..')
from util import get_data, plot_data
import scipy.optimize as spo
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality


def find_optimal_allocations(prices):
    """Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio.
    Parameters
    ----------
        prices: daily prices for each stock in portfolio
    Returns
    -------
        allocs: optimal allocations, as fractions that sum to 1.0
    """
    #tuple([(0,1)]*10)
    #constraints = ({ 'type': 'eq', 'fun': lambda inputs: 50.0 - np.sum(inputs) })

    num_columns = len(prices.columns)
    guesses = num_columns * [1. / num_columns, ]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(num_columns))
    opts = spo.minimize(min_func_sharpe, guesses, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)
    allocs = opts['x']
    return allocs

def min_func_sharpe(allocs, prices):
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(get_portfolio_value(prices, allocs, 1))
    return (-1*sharpe_ratio)

def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.
    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)
    Returns
    -------
        port_val: daily portfolio value
    """
    normed = prices/prices.ix[0,:]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.
    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)
    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    daily_ret = (port_val / port_val.shift(1)) - 1
    cum_ret = (port_val[-1] / port_val[0]) - 1
    std_daily_ret = daily_ret.std()
    avg_daily_ret = daily_ret.mean()
    k = np.sqrt(samples_per_year)
    sharpe_ratio = k * np.mean(avg_daily_ret - daily_rf) / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = np.asarray([0.2, 0.2, 0.3, 0.3]) # add code here to find the allocations
    allocs = find_optimal_allocations(prices)
    allocs = allocs / np.sum(allocs)

    # Get daily portfolio value
    port_val = prices_SPY  # add code here to compute daily portfolio values
    normalized = prices / prices.ix[0]
    allocated = normalized * allocs
    position_values = allocated
    port_val = position_values.sum(axis=1)

    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cr = (port_val.ix[-1] / port_val.ix[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(252) * (adr - 0) / sddr

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily portfolio value and SPY", ylabel="price", xlabel="Date")

    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
