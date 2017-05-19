"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
from datetime import datetime
# np.set_printoptions(threshold=np.nan)



def get_norm_data(prices):
    if isinstance(prices,pd.DataFrame): start_vals = prices.first('1D')
    else: start_vals = prices[0]
    normData = np.divide(prices, start_vals)

    return normData

def compute_portfolio_stats(port_values, rfr=0.0, sf=252.0):

    port_values = get_norm_data(port_values)
    port_daily_rets = np.divide(np.ediff1d(port_values), port_values[:-1])

    cr = port_values[-1] - 1.0
    adr = np.mean(port_daily_rets)
    sddr = np.std(port_daily_rets, ddof=1)

    sr = np.sqrt(sf) * (adr - rfr) / sddr

    return [cr, adr, sddr, sr]

def compute_portvals2(orders_file="train.csv", start_val=100000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    of= open(orders_file, 'w+')
    of.write('Date,Symbol,Order,Shares\n')
    if orders_file == "train.csv":
        of.write("2006-01-03,IBM,BUY,500\n")
        of.write("2009-12-31,IBM,SELL,500\n")
    else:
        of.write("2010-01-04,IBM,BUY,500\n")
        of.write("2010-12-31,IBM,SELL,500\n")
    of.close()

    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    os.remove(orders_file)
    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    orders_df.index = orders_df.index - start_date
    bad_index = datetime(2011, 6, 15) - start_date

    cos = orders_df['Symbol'].unique()
    portvals = get_data(cos, pd.date_range(start_date, end_date), addSPY=False)
    portMatrix = portvals.as_matrix()
    rows = np.isfinite(portMatrix[:, 0])

    Allocs = np.zeros((orders_df.index[-1].days + 1, len(cos)))
    Cash = np.zeros(orders_df.index[-1].days + 1)
    Cash.fill(100000)
    leverage = 0;  # (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
    stockVal = 0;
    for order in orders_df.iterrows():
        day = order[0].days
        sym = np.where(cos == order[1][0])[0][0]
        amt = order[1][2]
        if day == bad_index.days:
            continue
        if order[1][1] == 'BUY':
            Allocs[day][sym] += amt
            Cash[day:] -= amt * portMatrix[day][sym]
        else:
            Allocs[day][sym] -= amt
            Cash[day:] += amt * portMatrix[day][sym]

    Allocs = np.cumsum(Allocs, axis=0)
    norm_vals = np.sum(np.multiply(Allocs, portMatrix), axis=1);
    norm_vals = np.add(norm_vals, Cash)
    norm_vals = pd.DataFrame(data=norm_vals[rows], index=portvals.index[rows])
    return norm_vals


def compute_portvals(orders, start_date, end_date, start_val = 100000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    orders_file = open('orders.csv', 'w+')
    orders_file.write('Date,Symbol,Order,Shares\n')
    for order in orders.iterrows():
        # print order[1][1]
        orders_file.write(order[1][0].strftime('%Y-%m-%d') + ',' + order[1][1] + ',' + order[1][2] + ',' + str(order[1][3]) + '\n')
    orders_file.close()

    orders_df = pd.read_csv('orders.csv', index_col='Date', parse_dates=True, na_values=['nan'])

    # print orders_df
    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months

    orders_df.index = orders_df.index - start_date
    bad_index = datetime(2011, 6, 15) - start_date

    cos = orders_df['Symbol'].unique()
    portvals = get_data(cos, pd.date_range(start_date, end_date), addSPY=False)
    portMatrix = portvals.as_matrix()
    rows = np.isfinite(portMatrix[:, 0])
    portvals = portvals.fillna(method='ffill')
    portMatrix = portvals.as_matrix()

    Allocs = np.zeros((len(portvals), len(cos)))
    Cash = np.zeros(len(portvals))
    Cash.fill(start_val)
    leverage = 0;  # (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
    stockVal = 0;
    for order in orders_df.iterrows():
        day = order[0].days
        sym = np.where(cos == order[1][0])[0][0]
        amt = order[1][2]
        if day == bad_index.days:
            continue
        if order[1][1] == 'BUY':
            Allocs[day][sym] += amt
            Cash[day:] -= amt * portMatrix[day][sym]
        else:
            Allocs[day][sym] -= amt
            Cash[day:] += amt * portMatrix[day][sym]

    Allocs = np.cumsum(Allocs, axis=0)
    norm_vals = np.sum(np.multiply(Allocs, portMatrix), axis=1);
    norm_vals = np.add(norm_vals, Cash)
    norm_vals = pd.DataFrame(data=norm_vals[rows], index=portvals.index[rows])
    norm_vals.ix[0] = start_val

    return norm_vals


# def test_code():
#     # this is a helper function you can use to test your code
#     # note that during autograding his function will not be called.
#     # Define input parameters
#
#     of = "./orders/orders.csv"
#     sv = 1000000
#
#
#     # Get portfolio stats
#     # Here we just fake the data. you should use your code from previous assignments.
#     start_date = portvals.index[0]
#     end_date = portvals.index[-1]
#     cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(get_norm_data(portvals))
#     cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(
#                                                                             get_norm_data(
#                                                                                 get_data(['SPY'],pd.date_range(start_date,end_date)).as_matrix()))
#
#     # Compare portfolio against $SPX
#     print "Date Range: {} to {}".format(start_date, end_date)
#     print
#     print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
#     print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
#     print
#     print "Cumulative Return of Fund: {}".format(cum_ret)
#     print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
#     print
#     print "Standard Deviation of Fund: {}".format(std_daily_ret)
#     print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
#     print
#     print "Average Daily Return of Fund: {}".format(avg_daily_ret)
#     print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
#     print
#     print "Final Portfolio Value: {}".format(portvals[-1])
#
# if __name__ == "__main__":
#     test_code()
