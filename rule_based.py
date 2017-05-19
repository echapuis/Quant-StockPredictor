import indicators
import numpy as np
import pandas as pd
import datetime as dt
import math
import os
import time
import util
import sys
import helpers
import matplotlib.pyplot as plt
from cycler import cycler


# pd.set_option('display.height',2000)

def build_orders (data):

    price = data['Price']['IBM']
    sma = data['SMA']['IBM']
    smaRatio = data['SMA Ratio']['IBM']
    bbp = data['BBP']['IBM']
    ema = data['EMA']['IBM']
    emaRatio = data['EMA Ratio']['IBM']
    momentum = data['Momentum']['IBM']
    momentum = price.copy()
    momentum /= momentum.shift(10)
    momentum -= 1
    momentum = momentum.fillna(0)

    data_file = open('data.csv', 'w+')
    for i in range(len(data)):
        data_file.write(','.join(
            [str(price[i]), str(smaRatio[i]), str(bbp[i]), str(emaRatio[i]), str(momentum[i])]))
        data_file.write('\n')
    data_file.close()

    inf = open("data.csv")
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    # Orders starts as a NaN array of the same shape/index as price.
    orders = price.copy()
    orders.ix[:] = np.NaN

    sma_cross = smaRatio.copy()
    sma_cross[smaRatio >= 1] = 1
    sma_cross = sma_cross.diff()
    sma_cross.ix[0] = 0

    mom_cross = momentum.copy()
    mom_cross[momentum >= 1] = 1
    mom_cross = mom_cross.diff()
    mom_cross.ix[0] = 0

    # Apply our entry order conditions all at once.  This represents our TARGET SHARES
    # at this moment in time, not an actual order.


    orders[(bbp > 1.1) & (momentum > 0)] = -1
    orders[(bbp < -0.15) & (momentum < 0)] = 1
    orders[(smaRatio > 1) & (emaRatio > 1.05)] = 1
    orders[(smaRatio < 1) & (emaRatio < 0.9)] = -1

    # orders[momentum>0.15] = -1
    # orders[momentum<-0.2] = -1

    # Apply our exit order conditions all at once.  Again, this represents TARGET SHARES.
    # orders[(sma_cross != 0)] = 0

    # Forward fill NaNs with previous values, then fill remaining NaNs with 0.
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)

    # We now have a dataframe with our TARGET SHARES on every day, including holding periods.

    # Now take the diff, which will give us an order to place only when the target shares changed.
    orderChange = orders.diff()
    orderChange.ix[0] = 0

    # print orderChange
    # print type(orderChange)

    # And more importantly, drop all rows with no non-zero values (i.e. no orders).
    orderChange = orderChange.loc[(orderChange != 0)]

    # Now we have only the days that have orders.  That's better, at least!
    order_list = []

    # order_list.append([start_date.date(), 'IBM', 'BUY',0])
    size = orders.shape[0]
    lastTrade = -10
    curPos = 0
    for day in orderChange.index:
        loc = orders.index.get_loc(day)
        if loc < lastTrade + 10: continue
        if loc >= size - 10:
            break
        if orderChange.ix[day] > 0:
            if curPos == 1: continue
            order_list.append([day.date(), 'IBM', 'BUY', 500])
            lastTrade = loc
            curPos += 1
            if orders.iloc[loc + 10] > curPos:
                order_list.append([day.date()+dt.timedelta(days=14), 'IBM', 'BUY', 500])
                curPos += 1
        elif orderChange.ix[day] < 0:
            if curPos == -1: continue
            order_list.append([day.date(), 'IBM', 'SELL', 500])
            lastTrade = loc
            curPos -= 1
            if orders.iloc[loc + 10] < curPos:
                order_list.append([day.date()+dt.timedelta(days=14), 'IBM', 'SELL', 500])
                curPos -= 1


    # order_list.append([end_date.date(),'IBM','SELL',0])

    orders = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    os.remove("data.csv")

    return orders

def test_Manual(data='test', graph=True):
    test = data == 'test'
    if test:
        start_date = dt.datetime(2010, 01, 01)
        end_date = dt.datetime(2010, 12, 31)
        file = 'test.csv'
    else:
        start_date = dt.datetime(2006, 01, 01)
        end_date = dt.datetime(2009, 12, 31)
        file = 'train.csv'

    display = 'Momentum'
    graph_result = graph
    graph_extra = False
    symbols = ['IBM']
    # symbols = ['HD']

    lookback = 14
    startval = 100000

    data = indicators.get_indicators(symbols, start_date, end_date, lookback)

    orders = build_orders(data);

    portvals = helpers.compute_portvals(orders, start_date, end_date, startval)
    benchvals = helpers.compute_portvals2(file, startval)

    # print portvals

    prices = util.get_data(['IBM', 'SPY'], pd.date_range(start_date, end_date))

    norm_portvals = helpers.get_norm_data(portvals)
    norm_benchvals = helpers.get_norm_data(benchvals)
    # print prices
    # print norm_portvals
    # print norm_SPY

    if graph_result:
        plt.figure(0)
        df_temp = pd.concat([norm_portvals, norm_benchvals], keys=['Portfolio', 'Benchmark'], axis=1)
        df_temp.ix[0, 0] = 1
        df_temp['Portfolio'] = df_temp['Portfolio'].fillna(method='ffill')
        # print df_temp
        plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'black'])))
        plt.plot(df_temp)
        plt.legend(['Portfolio', 'Benchmark'], loc='upper left')
        plt.xticks(rotation=45)
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.grid()
        plt.title('Rule-based Portfolio vs. IBM')

        curPos = 'out'
        for order in orders.index:
            day = orders.ix[order, 'Date']
            ord = orders.ix[order, 'Order']
            if curPos == 'out' and ord == 'BUY':
                color = 'green'
                curPos = 'long'
            elif curPos == 'out' and ord == 'SELL':
                color = 'red'
                curPos = 'short'
            else:
                color = 'black'
                curPos = 'out'
            plt.axvline(day, color=color, linewidth=2)

    if graph_extra:
        plt.figure(1)
        df_temp = pd.concat([data[display]['IBM']], keys=['IBM'], axis=1)
        df_temp = df_temp.fillna(method='bfill')
        plt.plot(df_temp)
        plt.legend([display], loc='upper left')
        plt.xticks(rotation=45)
        plt.ylabel(display)
        plt.xlabel('Date')
        plt.grid()
        plt.title(display)
    plt.show()

    return norm_portvals, norm_benchvals

if __name__ == "__main__":
    test_Manual(data = 'train', graph=True)